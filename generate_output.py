from corrector import ScRNNChecker
import argparse
from utils import get_lines

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--task-name", dest="task_name", type=str)
parser.add_argument("--sentences-errors-file", dest="sentences_errors_file", type=str)
parser.add_argument("--sentences-predictions-file", dest="sentences_predictions_file", type=str)

params = vars(parser.parse_args())

language = params["language"]
task_name = params["task_name"]
sentences_errors_file = "data/" + params["sentences_errors_file"]
sentences_predictions_file = "output/" + params["sentences_predictions_file"]


def evaluate():
    sentences_errors_lines = get_lines(sentences_errors_file)
    checker = ScRNNChecker(language=language, task_name=task_name)
    sentences_prediction_lines = [checker.correct_string(sentence_errors_line) for sentence_errors_line in sentences_errors_lines]

    with open(sentences_predictions_file, "w") as f:
        for line in sentences_prediction_lines:
            f.write(f"{line}\n")


if __name__ == "__main__":
    evaluate()
