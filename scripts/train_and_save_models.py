import os
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from tqdm import tqdm

from models.simple_rnn import SimpleRNN
from models.train import train_model, prepare_optimizer_and_criterion
from models.utils import evaluate_text

def load_data(data_path='data/processed/shakespeare_vocab.pkl'):
    with open(data_path, 'rb') as f:
        token_ids, vocab, itos = pickle.load(f)
    return token_ids, vocab, itos

def create_data_loader(token_ids, batch_size=32, seq_length=50):
    # Prepare input-output sequences from the data
    data = []
    for i in tqdm(range(0, len(token_ids) - seq_length - 1)):
        input_seq = token_ids[i:i+seq_length]
        # The target is the next token, not the entire sequence
        target_token = token_ids[i+seq_length]
        data.append((torch.tensor(input_seq), torch.tensor(target_token)))

    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

def save_model(model, generation, model_dir='results/model_weights/'):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'gen_{generation}_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved: {model_path}')

def main():
    # Load the preprocessed data
    token_ids, vocab, itos = load_data()

    # Create a DataLoader
    data_loader = create_data_loader(token_ids)

    # Define model parameters
    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 256
    output_dim = vocab_size

    # Number of models to generate per generation
    num_models = 10
    num_generations = 5

    for generation in range(1, num_generations + 1):
        print(f"Starting Generation {generation}")

        for i in range(num_models):
            # Initialize the model
            model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim)

            # Prepare optimizer and criterion
            optimizer, criterion = prepare_optimizer_and_criterion(model)

            # Train the model
            train_model(model, data_loader, criterion, optimizer)

            # Save the model
            save_model(model, generation)

        print(f"Generation {generation} completed.\n")

if __name__ == '__main__':
    main()
