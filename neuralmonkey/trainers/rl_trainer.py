"""Training objectives for reinforcement learning."""

from typing import Callable

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.trainers.generic_trainer import Objective
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.logging import debug, log, warn
from neuralmonkey.vocabulary import END_TOKEN, PAD_TOKEN


# pylint: disable=invalid-name
RewardFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
# pylint: enable=invalid-name


def get_sequence_level_reward(references, hypotheses):
    rewards = [
        float(hypothesis == reference)
        for (reference, hypothesis) in zip(references, hypotheses)
    ]
    return rewards


def get_token_level_reward(references, hypotheses):
    rewards = []
    for reference, hypothesis in zip(references, hypotheses):
        reward = [
            float(len(reference) > i and hypothesis[i] == reference[i])
            for i in range(len(hypothesis))
        ]
        # For the end token, give 1.0 as reward if hypothesis and reference
        # have equal length.
        # reward.append(float(len(reference) == len(hypothesis)))
        # No, always give reward 0.0
        reward.append(0.0)
        rewards.append(reward)
    return rewards


# pylint: disable=too-many-locals
def rl_objective(decoder: Decoder,
                 reward_function: RewardFunction,
                 subtract_baseline: bool = False,
                 normalize: bool = False,
                 temperature: float = 1.,
                 ce_smoothing: float = 0.,
                 alpha: float = 1.,
                 sample_size: int = 1,
                 token_level: bool = False) -> Objective:
    """Construct RL objective for training with sentence-level feedback.

    Depending on the options the objective corresponds to:
    1) sample_size = 1, normalize = False, ce_smoothing = 0.0
     Bandit objective (Eq. 2) described in 'Bandit Structured Prediction for
     Neural Sequence-to-Sequence Learning'
     (http://www.aclweb.org/anthology/P17-1138)
     It's recommended to set subtract_baseline = True.
    2) sample_size > 1, normalize = True, ce_smoothing = 0.0
     Minimum Risk Training as described in 'Minimum Risk Training for Neural
     Machine Translation' (http://www.aclweb.org/anthology/P16-1159) (Eq. 12).
    3) sample_size > 1, normalize = False, ce_smoothing = 0.0
     The Google 'Reinforce' objective as proposed in 'Google’s NMT System:
     Bridging the Gap between Human and Machine Translation'
     (https://arxiv.org/pdf/1609.08144.pdf) (Eq. 8).
    4) sample_size > 1, normalize = False, ce_smoothing > 0.0
     Google's 'Mixed' objective in the above paper (Eq. 9),
     where ce_smoothing implements alpha.

    Note that 'alpha' controls the sharpness of the normalized distribution,
    while 'temperature' controls the sharpness during sampling.

    :param decoder: a recurrent decoder to sample from
    :param reward_function: any evaluator object
    :param subtract_baseline: avg reward is subtracted from obtained reward
    :param normalize: the probabilities of the samples are re-normalized
    :param sample_size: number of samples to obtain feedback for
    :param ce_smoothing: add cross-entropy loss with this coefficient to loss
    :param alpha: determines the shape of the normalized distribution
    :param temperature: the softmax temperature for sampling
    :return: Objective object to be used in generic trainer
    """
    check_argument_types()

    reference = decoder.train_inputs

    def _score_with_reward_function(references: np.array,
                                    hypotheses: np.array) -> np.array:
        """Score (time, batch) arrays with sentence-based reward function.

        Parts of the sentence after generated <pad> or </s> are ignored.
        BPE-postprocessing is also included.

        :param references: array of indices of references, shape (time, batch)
        :param hypotheses: array of indices of hypotheses, shape (time, batch)
        :return: an array of batch length with float rewards
        """
        rewards = []
        ref_sentences = []
        hyp_sentences = []
        for refs, hyps in zip(references.transpose(), hypotheses.transpose()):
            ref_seq = []
            hyp_seq = []
            for r_token in refs:
                token = decoder.vocabulary.index_to_word[r_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                ref_seq.append(token)
            for h_token in hyps:
                token = decoder.vocabulary.index_to_word[h_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                hyp_seq.append(token)
            # join BPEs, split on " " to prepare list for evaluator
            refs_tokens = " ".join(ref_seq).replace("@@ ", "").split(" ")
            hyps_tokens = " ".join(hyp_seq).replace("@@ ", "").split(" ")
            ref_sentences.append(refs_tokens)
            hyp_sentences.append(hyps_tokens)
        rewards = reward_function(ref_sentences, hyp_sentences)
        for i in range(min(len(rewards), 2)):
            debug('Ref: {ref}\nHyp: {hyp}\n Reward: {rew}'
                  .format(ref=' '.join(ref_sentences[i]),
                          hyp=' '.join(hyp_sentences[i]),
                          rew=rewards[i]))
        if token_level:
            # Pad rewards so that pad_token has reward 0
            max_len = max(map(len, rewards))
            #mask = np.stack([
            #    [1] * len(reward_v) + [0] * (max_len - len(reward_v))
            #    for reward_v in rewards
            #])
            rewards = np.stack([
                np.pad(
                    reward_v,
                    (0, max_len - len(reward_v)),
                    mode='constant',
                    constant_values=(0.0, 0.0)
                )
                for reward_v in rewards
            ])
            # Transpose so that rewards have shape (time, batch)
            rewards = rewards.transpose()
            # Sanity check
            if rewards.shape != hypotheses.shape:
                warn('Rewards and hypotheses have different shapes: {} and {}\n'
                     'Full rewards: {}\nFull hypotheses: {}'
                     .format(rewards.shape, hypotheses.shape, rewards, hypotheses))
            # Put mask into same array as rewards.
            # It will later be unpacked again.
            #rewards = (rewards, mask)
        return np.array(rewards, dtype=np.float32)

    samples_rewards = []
    samples_logprobs = []

    for _ in range(sample_size):
        # sample from logits
        # decoded, shape (time, batch)
        sample_loop_result = decoder.decoding_loop(train_mode=False,
                                                   sample=True,
                                                   temperature=temperature)
        sample_logits = sample_loop_result[0]
        sample_decoded = sample_loop_result[3]

        # rewards, shape (batch) for sequence level rewards
        # shape (time, batch) for token level rewards
        # simulate from reference
        sample_reward = tf.py_func(_score_with_reward_function,
                                   [reference, sample_decoded],
                                   tf.float32)
        #if token_level:
        #    sample_reward, sample_reward_mask = sample_reward

        # pylint: disable=invalid-unary-operand-type
        # Negative because we are doing gradient descent.
        word_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sample_decoded, logits=sample_logits)

        if token_level:
            logprobs = word_logprobs
        else:
            # sum word log prob to sentence log prob
            # no masking here, since otherwise shorter sentences are preferred
            sent_logprobs = tf.reduce_sum(word_logprobs, axis=0)
            logprobs = sent_logprobs

        # sample_size x batch or sample_size x time x batch
        samples_rewards.append(sample_reward)
        # sample_size x batch or sample_size x time x batch
        samples_logprobs.append(logprobs)

    # stack samples, sample_size x batch
    samples_rewards_stacked = tf.stack(samples_rewards)
    samples_logprobs_stacked = tf.stack(samples_logprobs)

    if subtract_baseline:
        # if specified, compute the average reward baseline
        reward_counter = tf.Variable(0.0, trainable=False,
                                     name="reward_counter")
        reward_sum = tf.Variable(0.0, trainable=False, name="reward_sum")
        # increment the cumulative reward
        reward_counter = tf.assign_add(
            reward_counter, tf.to_float(decoder.batch_size * sample_size))
        # sum over batch and samples
        reward_sum = tf.assign_add(reward_sum,
                                   tf.reduce_sum(samples_rewards_stacked))
        # compute baseline: avg of previous rewards
        baseline = tf.div(reward_sum,
                          tf.maximum(reward_counter, 1.0))
        samples_rewards_stacked -= baseline

        tf.summary.scalar(
            "train_{}/rl_reward_baseline".format(decoder.data_id),
            tf.reduce_mean(baseline), collections=["summary_train"])

    if normalize:
        # normalize over sample space
        samples_logprobs_stacked = tf.nn.softmax(
            samples_logprobs_stacked * alpha, dim=0)

    scored_probs = tf.stop_gradient(
        tf.negative(samples_rewards_stacked)) * samples_logprobs_stacked

    if token_level:
        # sum over time
        scored_probs = tf.reduce_sum(scored_probs, axis=1)

    # sum over samples
    total_loss = tf.reduce_sum(scored_probs, axis=0)

    # average over batch
    batch_loss = tf.reduce_mean(total_loss)

    if ce_smoothing > 0.0:
        batch_loss += tf.multiply(ce_smoothing, decoder.cost)

    tf.summary.scalar(
        "train_{}/self_rl_cost".format(decoder.data_id),
        batch_loss,
        collections=["summary_train"])

    return Objective(
        name="{}_rl".format(decoder.name),
        decoder=decoder,
        loss=batch_loss,
        gradients=None,
        weight=None
    )


def dpm_objective(decoder: Decoder, reweighing: bool = False) -> Objective:
    """Deterministic Propensity Matching

    See: http://www.aclweb.org/anthology/D/D17/D17-1272.pdf

    """
    check_argument_types()

    if decoder.feedback:
        log('DPM training with {}; reweighing: {}'
            .format(decoder.feedback, reweighing))
    else:
        msg = 'Decoder does not accept feedback'
        warn(msg)
        raise ValueError(msg)

    # Logged translation and corresponding logged rewards
    hypothesis = decoder.train_inputs  # time, batch
    rewards = decoder.train_rewards  # batch, time
    hypothesis = tf.Print(hypothesis, [hypothesis], "hypothesis", 10)
    rewards = tf.Print(rewards, [rewards], "rewards", 10)

    log('hypothesis: {}\nlogits: {}\nmask: {}\nrewards'
        .format(hypothesis.shape, decoder.train_logits,
                decoder.train_mask, rewards))
    # The minus makes cancels the minus from the cross entropy
    # so the result is + log p(y_i)
    # shape: batch, time
    word_logprobs = -tf.contrib.seq2seq.sequence_loss(
        decoder.train_logits,
        hypothesis,
        decoder.train_mask,
        average_across_timesteps=False,
        average_across_batch=False
    )
    #word_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
    #    labels=tf.transpose(hypothesis),
    #    logits=tf.transpose(decoder.train_logits, perm=[1, 0, 2])
    #)
    word_logprobs = tf.Print(word_logprobs, [word_logprobs], "word_logprobs", 10)
    sent_logprobs = tf.reduce_sum(word_logprobs, axis=0)
    sent_logprobs = tf.Print(sent_logprobs, [sent_logprobs], "sent_logprobs", 10)

    if decoder.feedback == 'token_level':
        # Negative rewards to make it a loss for using with gradient descent
        loss = tf.stop_gradient(tf.negative(rewards)) * tf.exp(word_logprobs)
        # Product over token-level losses calculated in log space
        zeros = tf.zeros_like(loss)
        loss = tf.where(loss > zeros, loss, zeros)
        loss = tf.exp(tf.reduce_sum(loss, axis=0))
    else:
        # Negative rewards to make it a loss for using with gradient descent
        loss = tf.stop_gradient(tf.negative(rewards)) * tf.exp(sent_logprobs)

    # Average over batch
    batch_loss = tf.reduce_mean(loss)
    batch_loss = tf.Print(batch_loss, [batch_loss], "batch_loss", 10)

    if reweighing:
        batch_loss /= tf.stop_gradient(tf.reduce_sum(sent_logprobs))

    return Objective(
        name="{}_dpm".format(decoder.name),
        decoder=decoder,
        loss=batch_loss,
        gradients=None,
        weight=None
    )
