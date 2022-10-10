from corrector import ScRNNChecker
import argparse
from utils import get_lines

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--sentences-errors-file", dest="sentences_errors_file", type=str)
parser.add_argument("--sentences-predictions-file", dest="sentences_predictions_file", type=str)

params = vars(parser.parse_args())

language = params["language"]
sentences_errors_file = "data/" + params["sentences_errors_file"]
sentences_predictions_file = "output/" + params["sentences_predictions_file"]


def evaluate():
    sentences_errors_lines = get_lines(sentences_errors_file)
    sentences_predictions = []
    checker_predictions = []

    # _teacher_general
    checker = ScRNNChecker(language=language, task_name="unnorm_teacher_general")
    predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
    checker_predictions.append(predictions)

    if language != "yi":
        checker = ScRNNChecker(language=language, task_name="phongrafamb_teacher_general")
        predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
        checker_predictions.append(predictions)

    checker = ScRNNChecker(language=language, task_name="keyprox_teacher_general")
    predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
    checker_predictions.append(predictions)

    checker = ScRNNChecker(language=language, task_name="random_teacher_general")
    predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
    checker_predictions.append(predictions)

    checker = ScRNNChecker(language=language, task_name="sylsim_teacher_general")
    predictions = [checker.correct_string(sentence_error_line) for sentence_error_line in sentences_errors_lines]
    checker_predictions.append(predictions)

    checker_predictions = [[row[i] for row in checker_predictions] for i in range(len(sentences_errors_lines))]

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

    with open(sentences_predictions_file, "w") as f:
        for line in sentences_predictions:
            f.write(f"{line}\n")


if __name__ == "__main__":
    evaluate()
