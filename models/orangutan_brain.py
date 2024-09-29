import torch.nn as nn

class OrangutanBrain(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, output_dim, num_layers=4):
        super(OrangutanBrain, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                   nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # Embed the tokens
        x = x.permute(1, 0, 2)  # Transformer expects (sequence_length, batch_size, embedding_dim)
        out = self.transformer(x)  # Pass through transformer
        out = self.fc(out[-1])  # Take the output at the last time step
        return out, None
