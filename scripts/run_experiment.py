import pickle
from models.simple_rnn import SimpleRNN
from models.evolve_population import evolve_population
from models.train import prepare_optimizer_and_criterion
from models.utils import evaluate_text

if __name__ == '__main__':
    with open('data/processed/shakespeare_vocab.pkl', 'rb') as f:
        token_ids, vocab, itos = pickle.load(f)

    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 256
    output_dim = vocab_size

    # Initialize population
    population_size = 20
    models = [SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim) for _ in range(population_size)]

    # Train each model initially (or as needed)
    for model in models:
        optimizer, criterion = prepare_optimizer_and_criterion(model)
        # Assuming you have a DataLoader `data_loader`
        train_model(model, data_loader, criterion, optimizer)

    # Evolve the population
    evolved_models = evolve_population(models, token_ids, evaluate_text)
