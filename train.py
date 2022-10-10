import numpy as np
import utils
from utils import TARGET_PAD_IDX, create_vocab, draw_result, get_batched_input_data, get_lines, get_vocab_size, load_vocab_dicts, set_word_limit
import argparse
import time
import torch
import sacrebleu
from torch import nn
from torch.autograd import Variable
from model import ScRNN

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--language", dest="language", type=str, default="shi")
parser.add_argument("--task-name", dest="task_name", type=str, default="random")
parser.add_argument("--vocab-size", dest="vocab_size", type=int, default=0)
parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=100)
parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)

parser.add_argument("--model-name", dest="model_name", type=str, default="shi_random_20531_100_32")
parser.add_argument("--new-vocab", dest="new_vocab", action="store_true")
parser.add_argument("--train", dest="need_to_train", action="store_true")
parser.add_argument("--train-sentences-errors-file", dest="train_sentences_errors_file", type=str, default="shi.random.train.sentences.errors.txt")
parser.add_argument("--train-sentences-file", dest="train_sentences_file", type=str, default="shi.random.train.sentences.txt")
parser.add_argument("--val-sentences-errors-file", dest="val_sentences_errors_file", type=str, default="shi.random.val.sentences.errors.txt")
parser.add_argument("--val-sentences-file", dest="val_sentences_file", type=str, default="shi.random.val.sentences.txt")
parser.add_argument("--test-sentences-errors-file", dest="test_sentences_errors_file", type=str, default="shi.random.test.sentences.errors.txt")
parser.add_argument("--test-sentences-file", dest="test_sentences_file", type=str, default="shi.random.test.sentences.txt")
parser.add_argument("--metric", dest="metric", type=str, default="WER")
parser.add_argument("--save", dest="save_model", action="store_true", default=False)

params = vars(parser.parse_args())

language = params["language"]
task_name = params["task_name"]
vocab_size = params["vocab_size"]
NUM_EPOCHS = params["num_epochs"]
batch_size = params["batch_size"]
# train files
train_sentences_errors_file = "data/" + params["train_sentences_errors_file"]
train_sentences_file = "data/" + params["train_sentences_file"]

# if vocab_size = 0, default all
if vocab_size == 0:
    vocab_size = get_vocab_size(train_sentences_file)


set_word_limit(vocab_size, language, task_name)
WORD_LIMIT = vocab_size
STOP_AFTER = 25
PWD = "/Users/gerardo/Documents/MaestriaPUCP/Tesis/Código/main/adversarial_misspellings_master/defenses/scRNN/"

# model paths
MODEL_PATH = "{}/models/{}".format(PWD, params["model_name"])

new_vocab = params["new_vocab"]
need_to_train = params["need_to_train"]


# path to vocabs
w2i_PATH = "{}vocab/{}_{}_w2i_{}.p".format(PWD, language, task_name, str(vocab_size))
i2w_PATH = "{}vocab/{}_{}_i2w_{}.p".format(PWD, language, task_name, str(vocab_size))
CHAR_VOCAB_PATH = "{}vocab/{}_{}_cv_{}.p".format(PWD, language, task_name, str(vocab_size))

# val/test files

val_sentences_errors_file = "data/" + params["val_sentences_errors_file"]
val_sentences_file = "data/" + params["val_sentences_file"]

test_sentences_errors_file = "data/" + params["test_sentences_errors_file"]
test_sentences_file = "data/" + params["test_sentences_file"]

metric = params["metric"]
save = params["save_model"]

# sanity check...
print("\n")
print("Parameters")
print(params)

"""
[Takes in predictions (y_preds) in integers, outputs a human readable
output line. In case when the prediction is UNK, it uses the input word as is.
Hence, input_line is also needed to know the corresponding input word.]
"""


def decode_line(input_line, y_preds):
    SEQ_LEN = len(input_line.split())
    assert (SEQ_LEN == len(y_preds))

    predicted_words = []
    for idx in range(SEQ_LEN):
        if y_preds[idx] == WORD_LIMIT:
            word = input_line.split()[idx]
        else:
            word = utils.i2w[y_preds[idx]]
        predicted_words.append(word)

    return " ".join(predicted_words)


"""
    [computes the word error rate]
    true_lines are what the model should have predicted, whereas
    output_lines are what the model ended up predicted
"""


def compute_WER(true_lines, output_lines):
    assert (len(true_lines) == len(output_lines))
    size = len(output_lines)

    error = 0.0
    total_words = 0.0

    for i in range(size):
        true_words = true_lines[i].split()
        output_words = output_lines[i].split()
        assert (len(true_words) == len(output_words))
        total_words += len(true_words)
        for j in range(len(output_words)):
            if true_words[j] != output_words[j]:
                error += 1.0

    return (100. * error/total_words)


def compute_chrF(true_lines, output_lines):
    assert (len(true_lines) == len(output_lines))
    size = len(output_lines)
    sum = 0

    for i in range(size):
        sum += sacrebleu.corpus_chrf([output_lines[i]], [[true_lines[i]]]).score

    return (sum/size)


def compute_BLEU(true_lines, output_lines):
    assert (len(true_lines) == len(output_lines))
    size = len(output_lines)
    sum = 0

    for i in range(size):
        sum += sacrebleu.corpus_bleu([output_lines[i]], [[true_lines[i]]]).score

    return (sum/size)


def iterate(model, optimizer, data_lines, data_error_lines, need_to_train, desc, iter_count, metric, print_stuff=True):
    sorted_lines = sorted(zip(data_lines, data_error_lines), key=lambda x: len(x[0].split()), reverse=True)
    data_lines, data_error_lines = zip(*sorted_lines)
    Xtype = torch.FloatTensor
    ytype = torch.LongTensor
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=TARGET_PAD_IDX)
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        Xtype = torch.cuda.FloatTensor
        ytype = torch.cuda.LongTensor
        criterion.cuda()

    predicted_lines = []
    true_lines = []
    total_loss = 0.0

    for input_lines, modified_lines, X, y, lens in get_batched_input_data(data_lines, data_error_lines, batch_size):
        true_lines.extend(input_lines)
        tx = Variable(torch.from_numpy(X)).type(Xtype)
        ty_true = Variable(torch.from_numpy(y)).type(ytype)

        ty_pred = model(tx, lens)
        y_pred = ty_pred.detach().cpu().numpy()

        for idx in range(len(input_lines)):
            y_pred_i = [np.argmax(y_pred[idx][:, i]) for i in range(lens[idx])]
            predicted_lines.append(decode_line(modified_lines[idx], y_pred_i))

        # compute loss
        loss = criterion(ty_pred, ty_true)
        total_loss += loss.item()

        if need_to_train:
            # backprop the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    average_loss = total_loss/len(true_lines)
    if metric == "WER":
        metric_value = compute_WER(true_lines, predicted_lines)
    elif metric == "chrF":
        metric_value = compute_chrF(true_lines, predicted_lines)
    elif metric == "BLEU":
        metric_value = compute_BLEU(true_lines, predicted_lines)

    if print_stuff:
        print("Average %s loss after %d iteration = %0.4f" % (desc, iter_count, average_loss))
        print("Total %s %s after %d iteration = %0.4f" % (desc, metric, iter_count, metric_value))

    return average_loss, metric_value


def main():
    train_sentences_lines = get_lines(train_sentences_file)
    train_sentences_errors_lines = get_lines(train_sentences_errors_file)

    print("\n")
    if new_vocab:
        print("Creating new vocabulary")
        create_vocab(train_sentences_file)
    else:
        print("Loading existing vocabulary")
        load_vocab_dicts(w2i_PATH, i2w_PATH, CHAR_VOCAB_PATH)

    print("Len of w2i: ", len(utils.w2i))
    print("Len of i2w: ", len(utils.i2w))
    print("Len of char vocab: ", len(utils.CHAR_VOCAB))

    print("\n")
    if need_to_train:
        print("Initializing an only ScRNN model")
        model = ScRNN(len(utils.CHAR_VOCAB), 50, WORD_LIMIT + 1)  # +1 for UNK
    else:
        model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=TARGET_PAD_IDX)
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        model.cuda()

    val_sentences_lines = get_lines(val_sentences_file)
    val_sentences_errors_lines = get_lines(val_sentences_errors_file)
    test_sentences_lines = get_lines(test_sentences_file)
    test_sentences_errors_lines = get_lines(test_sentences_errors_file)

    if need_to_train:
        # begin training ...
        print("\n")
        print("Training the model...\n")
        last_dumped_idx = 99999
        iterations = []
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []

        if metric == "WER":
            best_val_metric = 100.0
        elif metric == "chrF" or metric == "BLEU":
            best_val_metric = 0.0

        for ITER in range(NUM_EPOCHS):

            st_time = time.time()

            curr_train_loss, curr_train_metric = iterate(model, optimizer, train_sentences_lines, train_sentences_errors_lines, True, 'train', ITER+1, metric)

            curr_val_loss, curr_val_metric = iterate(model, None, val_sentences_lines, val_sentences_errors_lines, False, 'val', ITER+1, metric)

            iterate(model, None, test_sentences_lines, test_sentences_errors_lines, False, 'test', ITER+1, metric)

            iterations.append(ITER)
            train_losses.append(curr_train_loss)
            train_metrics.append(curr_train_metric)
            val_losses.append(curr_val_loss)
            val_metrics.append(curr_val_metric)
            model_name = "{}_{}_{}_{}_{}".format(language, task_name, str(vocab_size), str(NUM_EPOCHS), str(batch_size))

            if save:
                # check if the val WER improved?
                if metric == "WER":
                    improved = curr_val_metric < best_val_metric
                elif metric == "chrF" or metric == "BLEU":
                    improved = curr_val_metric > best_val_metric

                if improved:
                    last_dumped_idx = ITER+1
                    best_val_metric = curr_val_metric
                    # informative names for model dump files

                    print("Dumping after ", ITER + 1)
                    torch.save(model, "models/{}".format(model_name))

            # report the time taken per iteration for train + val + test
            # (+ often save)
            en_time = time.time()
            print("Time for the iteration %0.1f seconds" % (en_time - st_time))
            print("\n")

            # check if there hasn't been enough progress since last few iters
            if ITER > STOP_AFTER + last_dumped_idx:
                # i.e it is not improving since 'STOP_AFTER' number of iterations
                print("\n\nAborting since there hasn't been much progress")
                break

        draw_result(iterations, train_losses, val_losses, "loss", model_name)
        draw_result(iterations, train_metrics, val_metrics, metric, model_name)
    else:
        # just run the model on validation and test...
        # print (" *** running the model on val and test set *** ")

        st_time = time.time()

        train_WER = iterate(model, None, train_sentences_lines, train_sentences_errors_lines, False, "train", 0, metric, True)

        val_WER = iterate(model, None, val_sentences_lines, val_sentences_errors_lines, False, "val", 0, metric, True)

        test_WER = iterate(model, None, test_sentences_lines, test_sentences_errors_lines, False, "test", 0, metric, True)

        # report the time taken per iteration for val + test
        en_time = time.time()
        print("\n\nTime for the testing process = %0.1f seconds" % (en_time - st_time))
        # model_name = MODEL_PATH.split("/")[-1]
        # print(model_name + "\t" + str(val_WER) + "\t" + str(test_WER))


main()
