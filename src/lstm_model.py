# класс модели LSTM
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequencePredictionLSTM(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            padding_idx
    ):
        super().__init__()
        # слой эмбеддинга с входной размерной vocab_size и выходной embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # слой LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        # dropout, normalization
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        # линейный слой
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_weights(self):
        # напишите xavier инициализацию весов
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.RNN):
                for name, param in layer.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)


    def forward(self, x, lengths):
        embedded = self.embedding(x)
        # "запакуем" тензор embedded, используя pack_padded_sequence
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        # посчитайте выход rnn
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(output)