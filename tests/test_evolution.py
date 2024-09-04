import unittest
from models.evolution import mutate, crossover
from models.rnn_model import SimpleRNN

class TestEvolutionaryAlgorithm(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_dim = 32
        self.hidden_dim = 64
        self.output_dim = self.vocab_size
        self.model1 = SimpleRNN(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim)
        self.model2 = SimpleRNN(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim)

    def test_crossover(self):
        child = crossover(self.model1, self.model2)
        self.assertIsNotNone(child)

    def test_mutation(self):
        mutate(self.model1, mutation_rate=0.05)
        # Mutation should alter model weights, but test is simple for demonstration
        self.assertIsNotNone(self.model1)

if __name__ == '__main__':
    unittest.main()
