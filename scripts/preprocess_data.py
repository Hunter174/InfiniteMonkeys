import nltk
import os
import pickle

nltk.download('shakespeare')
from nltk.corpus import shakespeare


def preprocess_text(text):
    tokens = list(text)
    vocab = {c: i for i, c in enumerate(set(tokens))}
    itos = {i: c for c, i in vocab.items()}
    token_ids = [vocab[c] for c in tokens]
    return token_ids, vocab, itos


def save_preprocessed_data(data, vocab, itos, output_dir='data/processed/'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the preprocessed data
    with open(f'{output_dir}shakespeare_vocab.pkl', 'wb') as f:
        pickle.dump((data, vocab, itos), f)


if __name__ == '__main__':
    shakespeare_corpus = shakespeare.raw()
    token_ids, vocab, itos = preprocess_text(shakespeare_corpus)
    save_preprocessed_data(token_ids, vocab, itos)
