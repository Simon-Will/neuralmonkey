import os.path
from subprocess import Popen
import traceback
from typing import List

import mrl

from neuralmonkey.logging import debug, log, warn


class CountCorrectResult:

    def __init__(self, correct, total=None):
        self.correct = correct
        self.total = total

    @property
    def accuracy(self):
        if self.total is None:
            raise ValueError('Cannot calculate accuracy without knowing total')
        else:
            return self.correct / self.total


class AccuracyFromFunction:

    def __init__(self, count_correct, name='accuracy_from_function'):
        self.count_correct = count_correct
        self.name = name

    def __call__(self, decoded: List[List[str]],
                 references: List[List[str]]) -> float:
        total = len(decoded)
        try:
            count_correct_result = self.count_correct(decoded, references)
        except Exception as e:
            warn('Caught {} while evaluating. Proceeding anyway. The traceback'
                 ' was: {}'.format(type(e), traceback.format_exc()))
            return 0.0

        if isinstance(count_correct_result, int):
            accuracy = count_correct_result / total
        elif isinstance(count_correct_result, CountCorrectResult):
            if count_correct_result.total is None:
                accuracy = count_correct_result.correct / total
            else:
                accuracy = count_correct_result.accuracy
        else:
            raise TypeError('count_correct returned {} which is of type {},'
                            ' but expected type int or CountCorrectResult'
                            .format(count_correct_result,
                                    type(count_correct_result)))
            accuracy = 0.0

        return accuracy


def _get_line_count_in_file(filename):
    with open(filename) as f:
        line_count = sum(1 for line in f)
    return line_count

def make_nlmaps_counter(executable, db_dir, file_dir,
                        precomputed_references_answers=None):
    os.makedirs(file_dir, exist_ok=True)
    decoded_queries = os.path.join(file_dir, 'decoded_queries')
    references_queries = os.path.join(file_dir, 'references_queries')
    decoded_answers = os.path.join(file_dir, 'decoded_answers')
    references_answers = os.path.join(file_dir, 'references_answers')

    mrl_world = mrl.MRLS['nlmaps']()

    start_line = 0
    if precomputed_references_answers:
        precomputed_references_line_count = _get_line_count_in_file(
            precomputed_references_answers)

    def write_mrls_to_file(linearized_queries, filename):
        invalid_indices = []
        with open(filename, 'w') as f:
            for i, lin in enumerate(linearized_queries):
                mrl_ = mrl_world.functionalise(' '.join(lin))
                if mrl_:
                    f.write('{}\n'.format(mrl_))
                else:
                    f.write('INVALID QUERY\n')
                    invalid_indices.append(i)
        return invalid_indices


    def count_correct_with_precomputed(decoded, _):
        nonlocal start_line

        # Shadow references_answers from enclosing scope
        # since it is not needed in this function.
        references_answers = precomputed_references_answers

        batch_size = len(decoded)
        if batch_size == precomputed_references_line_count:
            log('Looks like validation time')
            reset_to_old_start_line = True
            old_start_line = start_line
            start_line = 0
            end_line = precomputed_references_line_count - 1
        else:
            log('Looks like logging time')
            reset_to_old_start_line = False
            if start_line + batch_size < precomputed_references_line_count:
                end_line = start_line + batch_size - 1
            else:
                end_line = precomputed_references_line_count - 1
                warn('Reached end of batch. start_line: {}, end_line: {},'
                     ' batch_size: {}, batch: {}'
                     .format(start_line, end_line, batch_size, decoded))

        if start_line == 0:
            # Report answers file once per evaluation.
            log('Using {} for the precomputed references answers'
                .format(references_answers))

        debug('Evaluating lines {} to {} in file {}'
              .format(start_line, end_line, references_answers))

        # Translate queries from linearzied to functional MRL.
        write_mrls_to_file(decoded, decoded_queries)

        # Get answers for valid queries.
        decoded_args = [executable, '-d', db_dir, '-f', decoded_queries,
                        '-a', decoded_answers]
        decoded_proc = Popen(decoded_args)

        # Wait for the job to finish.
        decoded_status = decoded_proc.wait()
        msg = ('MRL execution on decoded exited with status {}.'
               .format(decoded_status))
        if decoded_status != 0:
            warn(msg)
        else:
            debug(msg)

        expected_decoded_line_count = end_line - start_line + 1
        observed_decoded_line_count = _get_line_count_in_file(decoded_answers)
        if observed_decoded_line_count != expected_decoded_line_count:
            warn('Expected line count in decoded answers to be {} - {} + 1 ='
                 ' {}, but it is {}'
                 .format(end_line, start_line, expected_decoded_line_count,
                         observed_decoded_line_count))

        with open(decoded_answers) as decoded_f, \
             open(references_answers) as references_f:
            correct = 0
            for cur_line, reference_answer in enumerate(references_f):
                if start_line <= cur_line <= end_line:
                    decoded_answer = next(decoded_f)
                    #debug('Comparing decoded answer {} to reference answer {}'
                    #      .format(decoded_answer, reference_answer))
                    if decoded_answer == reference_answer:
                        correct += 1

        # Set start line for next batch.
        if reset_to_old_start_line:
            start_line = old_start_line
        else:
            start_line = (end_line + 1
                          if end_line + 1 < precomputed_references_line_count
                          else 0)
        debug('{} out of {} correct'.format(correct, batch_size))
        return CountCorrectResult(correct, batch_size)


    def count_correct_without_precomputed(decoded, references):
        if len(decoded) != len(references):
            warn('Length of decoded ({}) does not match length of references'
                 ' ({})'.format(len(decoded), len(references)))

        # Translate decoded queries from linearzied to functional MRL.
        write_mrls_to_file(decoded, decoded_queries)
        # Get answers for valid queries.
        decoded_args = [executable, '-d', db_dir, '-f', decoded_queries,
                        '-a', decoded_answers]
        decoded_proc = Popen(decoded_args)

        # Translate reference queries from linearzied to functional MRL.
        write_mrls_to_file(references, references_queries)
        references_args = [executable, '-d', db_dir, '-f', references_queries,
                           '-a', references_answers]
        references_proc = Popen(references_args)

        # Wait for the jobs to finish.
        references_status = references_proc.wait()
        decoded_status = decoded_proc.wait()

        msg = ('MRL execution on decoded and references exited with status'
               ' codes {} and {}, respectively.'
               .format(decoded_status, references_status))
        if decoded_status == 0 and references_status == 0:
            debug(msg)
        else:
            warn(msg)

        with open(decoded_answers) as decoded_f, \
             open(references_answers) as references_f:
            correct = sum(
                decoded_answer == references_answer
                for decoded_answer, reference_answer
                in zip(decoded_f, references_f)
            )
        return CountCorrectResult(correct)

    if precomputed_references_answers:
        count_correct_fn = count_correct_with_precomputed
    else:
        count_correct_fn = count_correct_without_precomputed
    return count_correct_fn


def make_exact_match_counter():
    def count_matches(decoded, references):
        return sum(decoded_lin == references_lin
                   for decoded_lin, references_lin
                   in zip(decoded, references))
    return count_matches
