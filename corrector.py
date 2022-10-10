""" class using Semi Character RNNs as a defense mechanism
    ScRNN paper: https://arxiv.org/abs/1608.02214
"""
import os
import utils
from utils import *  # FIXME: this shouldn't be this way

# torch related imports
import torch
from torch import nn
from torch.autograd import Variable

# elmo related imports
from allennlp.modules.elmo import batch_to_ids


class ScRNNChecker(object):
    def __init__(self, language, task_name):
        PWD = os.path.dirname(os.path.realpath(__file__))

        if language == "shi" or language == "ash" or language == "ya":
            vocab_size = 4999
        else:
            vocab_size = 2999

        self.vocab_size = vocab_size
        MODEL_PATH = "{}/models/{}_{}_{}_100_32".format(PWD, language, task_name, vocab_size)
        print(MODEL_PATH)

        # path to vocabs
        w2i_PATH = "{}/vocab/{}_{}_w2i_{}.p".format(PWD, language, task_name, vocab_size)
        i2w_PATH = "{}/vocab/{}_{}_i2w_{}.p".format(PWD, language, task_name, vocab_size)
        CHAR_VOCAB_PATH = "{}/vocab/{}_{}_cv_{}.p".format(PWD, language, task_name, vocab_size)

        set_word_limit(int(vocab_size), language, task_name)
        load_vocab_dicts(w2i_PATH, i2w_PATH, CHAR_VOCAB_PATH)

        self.model = torch.load(MODEL_PATH, map_location="cpu")
        self.predicted_unks = 0.0
        self.predicted_unks_in_vocab = 0.0
        self.total_predictions = 0.0

        return

    def correct_string(self, line):
        line = line.strip().lower()
        Xtype = torch.FloatTensor
        ytype = torch.LongTensor
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            self.model.cuda()
            Xtype = torch.cuda.FloatTensor
            ytype = torch.cuda.LongTensor

        X, _ = get_line_representation(line, line)
        tx = Variable(torch.from_numpy(np.array([X]))).type(Xtype)

        SEQ_LEN = len(line.split())
        ty_pred = self.model(tx, [SEQ_LEN])
        y_pred = ty_pred.detach().cpu().numpy()
        y_pred = y_pred[0]  # ypred now is NUM_CLASSES x SEQ_LEN

        output_words = []
        self.total_predictions += SEQ_LEN

        for idx in range(SEQ_LEN):
            pred_idx = np.argmax(y_pred[:, idx])
            # print(pred_idx)
            if pred_idx == utils.WORD_LIMIT:
                word = line.split()[idx]
                output_words.append(word)
                self.predicted_unks += 1.0
                if word in utils.w2i:
                    self.predicted_unks_in_vocab += 1.0
            else:
                output_words.append(utils.i2w[pred_idx])

        return " ".join(output_words)
