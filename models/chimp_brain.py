import torch.nn as nn

class ChimpBrain(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2):
        super(ChimpBrain, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 because of bidirectional

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, (hidden, cell) = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out, (hidden, cell)
