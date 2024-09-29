import torch.nn.functional as F
import os
import pickle
from models.simple_monkey import MonkeyBrain
import random
import torch
import copy


def generate_text(model, seed_text="To be or not to be", max_length=100, vocab=None, itos=None):
    model.eval()  # Set the model to evaluation mode
    generated_text = seed_text

    # Convert seed text to indices using the `vocab`
    input_seq = torch.tensor([vocab[c] for c in seed_text]).unsqueeze(0)
    hidden = None  # Initialize hidden state as None for the first iteration

    for _ in range(max_length):
        # Forward pass through the model
        logits, hidden = model(input_seq, hidden)

        # Print logits for debugging
        # print(logits)

        # Get the predicted next character (argmax over vocabulary logits)
        next_char = torch.argmax(logits, dim=1)

        # Debugging: Print the predicted next character's index
        # print(next_char)

        # Ensure `itos` is not None, and map the predicted index to a character
        if itos is not None:
            generated_text += itos[next_char.item()]
        else:
            raise ValueError("The `itos` mapping is not provided.")

        # Prepare input for the next iteration (using the predicted character)
        input_seq = next_char.unsqueeze(0)

    return generated_text

def calculate_perplexity(logits, target):

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    return torch.exp(loss).item()  # Convert loss to perplexity


# In models/utils.py, in the evaluate_text function
def evaluate_text(generated_text, vocab):
    # Normalize text case to match vocab keys (if vocab uses lowercase tokens)
    generated_tokens = generated_text.lower().split()

    # Handle missing tokens by mapping to a special token or skipping
    generated_ids = torch.tensor([vocab[token] for token in generated_tokens if token in vocab])

    return generated_ids


def load_data(data_path='data/processed/shakespeare_vocab.pkl'):
    with open(data_path, 'rb') as f:
        token_ids, vocab, itos = pickle.load(f)
    return token_ids, vocab, itos

def load_models(generation, num_models, vocab_size, embedding_dim, hidden_dim, output_dim):
    models = []
    for i in range(num_models):
        model = MonkeyBrain(vocab_size, embedding_dim, hidden_dim, output_dim)
        model_path = f'results/model_weights/gen_{generation}_model_{i}.pth'
        model.load_state_dict(torch.load(model_path, weights_only=True))

        models.append(model)
    return models

def evolve_and_save_models(models, generation, token_ids, vocab, itos, num_models=10):
    # Pass both `vocab` and `itos` to `evolve_population`
    evolved_models = evolve_population(models, token_ids, vocab, itos, evaluate_text)
    for i, model in enumerate(evolved_models):
        save_model(model, generation + 1, i)

def save_model(model, generation, model_num, model_dir='results/model_weights/'):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'gen_{generation}_model_{model_num}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}\n")



def evolve_population(models, corpus, vocab, itos, fitness_fn, num_generations=10, top_k=5, mutation_rate=0.01):
    for generation in range(num_generations):
        print(f"Generation {generation + 1}")

        # Evaluate fitness of each model
        fitness_scores = []
        for model in models:
            generated_text = generate_text(model, vocab=vocab, itos=itos)
            fitness = fitness_fn(generated_text, vocab)

            # Only append if the tensor is not empty
            if fitness.numel() > 0:  # numel() checks the number of elements in the tensor
                fitness_scores.append((fitness, model))

        # Ensure there's at least one valid model with a non-empty tensor
        if not fitness_scores:
            raise ValueError("All models produced empty fitness scores.")

        fitness_scores.sort(key=lambda x: x[0])
        top_models = [model for _, model in fitness_scores[:top_k]]

        # Create next generation via crossover and mutation
        next_generation = []
        for _ in range(len(models)):
            parent1, parent2 = random.sample(top_models, 2)
            child_model = crossover(parent1, parent2)
            mutate(child_model, mutation_rate)
            next_generation.append(child_model)

        models = next_generation

    return models

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    with torch.no_grad():
        for param1, param2 in zip(parent1.parameters(), parent2.parameters()):
            param1.copy_(0.5 * (param1 + param2))  # Average the parameters for crossover
    return child

def mutate(model, mutation_rate=0.01):
    with torch.no_grad():
        for param in model.parameters():
            if torch.rand(1).item() < mutation_rate:
                param.add_(torch.randn(param.size()) * mutation_rate)  # Apply mutation

