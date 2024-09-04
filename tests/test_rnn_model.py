import unittest
import torch
from models.rnn_model import SimpleRNN

class TestSimpleRNN(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_dim = 32
        self.hidden_dim = 64
        self.output_dim = self.vocab_size
        self.model = SimpleRNN(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim)
        self.input = torch.randint(0, self.vocab_size, (1, 10))

    def test_forward(self):
        output, hidden = self.model(self.input)
        self.assertEqual(output.shape[1], self.output_dim)

if __name__ == '__main__':
    unittest.main()
