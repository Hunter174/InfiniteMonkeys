import torch.nn as nn

class GorillaBrain(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GorillaBrain, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, (hidden, cell) = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out, (hidden, cell)
