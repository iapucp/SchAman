from collections import defaultdict
import numpy as np
import pickle
import matplotlib.pyplot as plt

CHAR_VOCAB = []
w2i = defaultdict(lambda: 0.0)
i2w = defaultdict(lambda: "UNK")

language = ""
WORD_LIMIT = 9999  # remaining 1 for <PAD> (this is inclusive of UNK)
task_name = ""
TARGET_PAD_IDX = -1
INPUT_PAD_IDX = 0


def set_word_limit(word_limit, lan, task):
    global language
    global WORD_LIMIT
    global task_name

    language = lan
    WORD_LIMIT = word_limit
    task_name = task


def get_lines(filename):
    f = open(filename)
    lines = f.readlines()
    lines = [line.strip().lower() for line in lines]

    return lines


def get_vocab_size(filename):
    lines = get_lines(filename)
    words = set(lines)
    vocab_size = len(words)

    return vocab_size


def create_vocab(filename):
    global w2i, i2w, CHAR_VOCAB
    lines = get_lines(filename)
    for line in lines:
        for word in line.split():
            # add all its char in vocab
            for char in word:
                if char not in CHAR_VOCAB:
                    CHAR_VOCAB.append(char)

            w2i[word] += 1.0

    word_list = sorted(w2i.items(), key=lambda x: x[1], reverse=True)
    word_list = word_list[:WORD_LIMIT]  # only need top few words

    # remaining words are UNKs ... sorry!
    w2i = defaultdict(lambda: WORD_LIMIT)  # default id is UNK ID
    w2i['<PAD>'] = INPUT_PAD_IDX  # INPUT_PAD_IDX is 0
    i2w[INPUT_PAD_IDX] = '<PAD>'
    for idx in range(WORD_LIMIT-1):
        w2i[word_list[idx][0]] = idx+1
        i2w[idx+1] = word_list[idx][0]

    pickle.dump(dict(w2i), open("vocab/{}_{}_w2i_{}.p".format(language, task_name, str(WORD_LIMIT)), "wb"))
    pickle.dump(dict(i2w), open("vocab/{}_{}_i2w_{}.p".format(language, task_name, str(WORD_LIMIT)), "wb"))
    pickle.dump(CHAR_VOCAB, open("vocab/{}_{}_cv_{}.p".format(language, task_name, str(WORD_LIMIT)), "wb"))


def load_vocab_dicts(wi_path, iw_path, cv_path):
    wi = pickle.load(open(wi_path, "rb"))
    iw = pickle.load(open(iw_path, "rb"))
    cv = pickle.load(open(cv_path, "rb"))

    convert_vocab_dicts(wi, iw, cv)


""" converts vocabulary dictionaries into defaultdicts
"""


def convert_vocab_dicts(wi, iw, cv):
    global w2i, i2w, CHAR_VOCAB
    CHAR_VOCAB = cv
    w2i = defaultdict(lambda: WORD_LIMIT)

    for w in wi:
        w2i[w] = wi[w]

    for i in iw:
        i2w[i] = iw[i]
    return


def get_target_representation(line):
    return [w2i[word] for word in line.split()]


def pad_input_sequence(X, max_len):
    assert (len(X) <= max_len)
    while len(X) != max_len:
        X.append([INPUT_PAD_IDX for _ in range(len(X[0]))])

    return X


def pad_target_sequence(y, max_len):
    assert (len(y) <= max_len)
    while len(y) != max_len:
        y.append(TARGET_PAD_IDX)

    return y


def get_batched_input_data(lines, lines_with_errors, batch_size):
    output = []
    for batch_start in range(0, len(lines), batch_size):
        batch_end = min(len(lines), batch_start + batch_size)
        input_lines = []
        modified_lines = []
        X = []
        y = []
        lens = []
        max_len = max([len(line.split()) for line in lines[batch_start: batch_end]])

        for line, line_with_errors in zip(lines[batch_start: batch_end], lines_with_errors[batch_start: batch_end]):
            X_i, modified_line_i = get_line_representation(line, line_with_errors)
            assert (len(line.split()) == len(modified_line_i.split()))
            y_i = get_target_representation(line)
            # pad X_i, and y_i
            X_i = pad_input_sequence(X_i, max_len)
            y_i = pad_target_sequence(y_i, max_len)
            # append input lines, modified lines, X_i, y_i, lens
            input_lines.append(line)
            modified_lines.append(modified_line_i)
            X.append(X_i)
            y.append(y_i)
            lens.append(len(modified_line_i.split()))

        output.append((input_lines, modified_lines, np.array(X), np.array(y), lens))

    return output


def get_line_representation(line, lines_with_errors):
    rep = []
    modified_words = []

    for word, word_with_error in zip(line.split(), lines_with_errors.split()):
        word_rep, new_word = get_word_representation(word, word_with_error)

        rep.append(word_rep)
        modified_words.append(new_word)

    return rep, " ".join(modified_words)


def get_word_representation(word, word_error):

    # dirty case
    if len(word) == 1 or len(word) == 2:
        rep = one_hot(word[0]) + zero_vector() + one_hot(word[-1])
        return rep, word

    #rep = one_hot(word[0]) + bag_of_chars(word[1:-1]) + one_hot(word[-1])
    rep = one_hot(word_error[0]) + bag_of_chars(word_error[1:-1]) + one_hot(word_error[-1])

    return rep, word_error


def one_hot(char):
    return [1.0 if ch == char else 0.0 for ch in CHAR_VOCAB]


def bag_of_chars(chars):
    return [float(chars.count(ch)) for ch in CHAR_VOCAB]


def zero_vector():
    return [0.0 for _ in CHAR_VOCAB]


def draw_result(x, y_train, y_val, y_axis_title, model_name):
    plt.plot(x, y_train, '-b', label="train")
    plt.plot(x, y_val, '-r', label="val")

    plt.locator_params(axis="x", nbins=10)
    plt.locator_params(axis="y", nbins=20)

    plt.xlabel("epoch")
    plt.ylabel(y_axis_title)
    plt.legend(loc="upper left")
    plt.grid()
    plt.title("{}: ({})".format(model_name, y_axis_title))

    plt.savefig("curves/{}_{}.png".format(model_name, y_axis_title))
    plt.show()
