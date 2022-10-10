from corrector import ScRNNChecker
import argparse
import sacrebleu
from utils import get_lines

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--sentences-errors-file", dest="sentences_errors_file", type=str)
parser.add_argument("--sentences-file", dest="sentences_file", type=str)

params = vars(parser.parse_args())

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

    chrF = compute_chrF(sentences_lines, sentences_errors_lines)

    print("chrF: {:.4f}".format(chrF))


if __name__ == "__main__":
    evaluate()
