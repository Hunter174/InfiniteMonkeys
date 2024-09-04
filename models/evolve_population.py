import random
import torch
import copy

def evolve_population(models, corpus, fitness_fn, num_generations=10, top_k=5, mutation_rate=0.01):
    for generation in range(num_generations):
        print(f"Generation {generation + 1}")

        # Evaluate fitness of each model
        fitness_scores = []
        for model in models:
            generated_text = generate_text(model, corpus)
            fitness = fitness_fn(generated_text, corpus)
            fitness_scores.append((fitness, model))

        # Sort models by fitness score (lower is better)
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
            param1.copy_(0.5 * (param1 + param2))
    return child

def mutate(model, mutation_rate=0.01):
    with torch.no_grad():
        for param in model.parameters():
            if torch.rand(1).item() < mutation_rate:
                param.add_(torch.randn(param.size()) * mutation_rate)
