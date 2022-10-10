from attr import NOTHING
from spacy import language
from corrector import ScRNNChecker
import argparse
import sacrebleu
from utils import get_lines

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--test-sentences-errors-file", dest="sentences_errors_file", type=str)
parser.add_argument("--test-sentences-file", dest="sentences_file", type=str)

params = vars(parser.parse_args())

language = params["language"]
sentences_errors_file = "data/" + params["sentences_errors_file"]
sentences_file = "data/" + params["sentences_file"]


def compute_chrF(true_lines, output_lines):
    assert (len(true_lines) == len(output_lines))
    size = len(output_lines)
    sum = 0

    for i in range(size):
        sum += sacrebleu.corpus_chrf([output_lines[i]], [[true_lines[i]]]).score

    return (sum/size)


def evaluate():
    sentences_errors_lines = get_lines(sentences_errors_file)
    sentences_lines = get_lines(sentences_file)
    sentences_predictions = []
    checker_predictions = []

    # _teacher_general
    checker = ScRNNChecker(language=language, task_name="unnorm")
    predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
    checker_predictions.append(predictions)

    if language != "yi":
        checker = ScRNNChecker(language=language, task_name="phongrafamb")
        predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
        checker_predictions.append(predictions)

    checker = ScRNNChecker(language=language, task_name="keyprox")
    predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
    checker_predictions.append(predictions)

    checker = ScRNNChecker(language=language, task_name="random")
    predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
    checker_predictions.append(predictions)

    checker = ScRNNChecker(language=language, task_name="sylsim")
    predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
    checker_predictions.append(predictions)

    checker_predictions = [[row[i] for row in checker_predictions] for i in range(len(sentences_lines))]

    for predictions in checker_predictions:
        word_length = len(predictions[0].split(" "))
        words_prediction_voting = []
        for idx in range(word_length):
            voting_dict = {}
            for sentence_prediction in predictions:
                word = sentence_prediction.split(" ")[idx]
                if word not in voting_dict:
                    voting_dict[word] = 1
                else:
                    voting_dict[word] += 1

            sorted_voting_dict = sorted(voting_dict, key=voting_dict.get, reverse=True)
            words_prediction_voting.append(list(sorted_voting_dict)[0])

        sentence_predicted_voting = " ".join(words_prediction_voting)
        sentences_predictions.append(sentence_predicted_voting)

    chrF = compute_chrF(sentences_lines, sentences_predictions)
    print("chrF: {:.4f}".format(chrF))


if __name__ == "__main__":
    evaluate()
