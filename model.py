from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ScRNN(nn.Module):
    def __init__(self, char_vocab_size, hdim, output_dim):
        super(ScRNN, self).__init__()
        """ layers """
        self.lstm = nn.LSTM(3*char_vocab_size, hdim, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*hdim, output_dim)

    """ size(inp) --> BATCH_SIZE x MAX_SEQ_LEN x EMB_DIM 
    """

    def forward(self, inp, lens):
        packed_input = pack_padded_sequence(inp, lens, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        h, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.linear(h)  # out is batch_size x max_seq_len x class_size
        out = out.transpose(dim0=1, dim1=2)

        return out  # out is batch_size  x class_size x  max_seq_len
