import os
import pickle
import torch
import random
from models.simple_monkey import MonkeyBrain
from models.utils import generate_text

def load_data(data_path='data/processed/shakespeare_vocab.pkl'):
    with open(data_path, 'rb') as f:
        token_ids, vocab, itos = pickle.load(f)
    return token_ids, vocab, itos

def load_model(model_path, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=None):
    model = MonkeyBrain(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == '__main__':
    # Load vocab and token information
    token_ids, vocab, itos = load_data()

    # Define the path to the model weights directory
    model_dir = 'results/model_weights'

    # Group models by generation
    generation_dict = {}
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.pth'):
            gen_num = file_name.split('_')[1]  # Assuming filename format is like gen_5_model_0.pth
            if gen_num not in generation_dict:
                generation_dict[gen_num] = []
            generation_dict[gen_num].append(file_name)

    # Iterate over each generation and pick one random model
    for gen, models in generation_dict.items():
        selected_model = random.choice(models)
        model_path = os.path.join(model_dir, selected_model)
        print(f"Loading model from {model_path}")

        # Load the selected model
        model = load_model(model_path, vocab_size=len(vocab), output_dim=len(vocab))

        # Generate and print text
        generated_text = generate_text(model, seed_text="To be or not to be", max_length=200, vocab=vocab, itos=itos)
        print(f"Generated text for {selected_model}:\n{generated_text}\n")
