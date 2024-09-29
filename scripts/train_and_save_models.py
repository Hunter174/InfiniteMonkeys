import os
import torch
import pickle
import shutil
import random
import numpy as np
from tqdm import tqdm
from models.simple_monkey import MonkeyBrain
from models.train import train_model, prepare_optimizer_and_criterion


def load_data(data_path='data/processed/shakespeare_vocab.pkl'):
    with open(data_path, 'rb') as f:
        token_ids, vocab, itos = pickle.load(f)
    return token_ids, vocab, itos


def create_data_loader(token_ids, batch_size=32, seq_length=50, max_samples=None):
    data = []
    total_sequences = len(token_ids) - seq_length - 1
    max_samples = min(max_samples, total_sequences) if max_samples else total_sequences

    for i in tqdm(range(max_samples)):
        input_seq = token_ids[i:i + seq_length]
        target_token = token_ids[i + seq_length]
        data.append((torch.tensor(input_seq), torch.tensor(target_token)))

    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


def save_model(model, generation, model_num, model_dir='results/model_weights/'):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'gen_{generation}_model_{model_num}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved: {model_path}')


def evaluate_model(model, data_loader, criterion):
    """Evaluate the model using validation data to compute fitness (loss)."""
    model.eval()  # Set to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            output, _ = model(inputs)
            loss = criterion(output, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)  # Compute average loss
    return avg_loss  # Lower loss means better fitness


def crossover(model1, model2):
    """Perform crossover between two models by averaging their weights."""
    child_model = MonkeyBrain(vocab_size=model1.embedding.num_embeddings, embedding_dim=model1.embedding.embedding_dim,
                              hidden_dim=model1.rnn.hidden_size, output_dim=model1.fc.out_features)
    with torch.no_grad():
        for param1, param2, child_param in zip(model1.parameters(), model2.parameters(), child_model.parameters()):
            child_param.copy_(0.5 * (param1 + param2))  # Average parameters
    return child_model


def mutate(model, mutation_rate=0.01):
    """Apply random mutations to a model's parameters."""
    with torch.no_grad():
        for param in model.parameters():
            if torch.rand(1).item() < mutation_rate:  # Apply mutation with some probability
                param.add_(torch.randn(param.size()) * mutation_rate)  # Add small noise to weights


def main():
    # Define model directory
    model_dir = 'results/model_weights/'

    # Remove the model directory if it exists
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print(f"Deleted existing directory: {model_dir}")

    # Load data and prepare data loader
    token_ids, vocab, itos = load_data()
    data_loader = create_data_loader(token_ids, batch_size=32, seq_length=50, max_samples=50000)

    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 256
    output_dim = vocab_size

    num_models = 5
    num_generations = 10
    top_k = 2  # Select the top 2 models for crossover and mutation

    # Criterion for evaluating models
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize population for the first generation
    population = []
    for i in range(num_models):
        model = MonkeyBrain(vocab_size, embedding_dim, hidden_dim, output_dim)
        optimizer, criterion = prepare_optimizer_and_criterion(model)
        train_model(model, data_loader, criterion, optimizer)
        population.append(model)

    for generation in range(1, num_generations + 1):
        print(f"Starting Generation {generation}")
        print("=" * 10)

        # Evaluate fitness of models (lower loss is better)
        fitness_scores = []
        for i, model in enumerate(population):
            loss = evaluate_model(model, data_loader, criterion)
            fitness_scores.append((loss, model))  # (loss, model) -> minimize loss

        # Sort by fitness score (lower loss is better)
        fitness_scores.sort(key=lambda x: x[0])
        print(f"Top models from Generation {generation} by loss: {[x[0] for x in fitness_scores[:top_k]]}")

        # Select top-k models
        top_models = [model for _, model in fitness_scores[:top_k]]

        # Create next generation via crossover and mutation
        next_generation = []
        for i in range(num_models):
            # Perform crossover between two random top models
            parent1, parent2 = random.sample(top_models, 2)
            child_model = crossover(parent1, parent2)

            # Mutate the child model
            mutate(child_model, mutation_rate=0.01)

            # Train the child model
            optimizer, criterion = prepare_optimizer_and_criterion(child_model)
            train_model(child_model, data_loader, criterion, optimizer)

            # Save the child model and add to the next generation
            save_model(child_model, generation, i, model_dir=model_dir)
            next_generation.append(child_model)

        # Replace the old population with the new generation
        population = next_generation

        print(f"Generation {generation} completed.\n")


if __name__ == '__main__':
    main()
