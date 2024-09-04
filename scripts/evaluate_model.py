import pickle
from models.rnn_model import SimpleRNN
from models.utils import generate_text

if __name__ == '__main__':
    # Load preprocessed data and model
    with open('data/processed/shakespeare_vocab.pkl', 'rb') as f:
        token_ids, vocab, itos = pickle.load(f)

    model = SimpleRNN(vocab_size=len(vocab), embedding_dim=128, hidden_dim=256, output_dim=len(vocab))
    model.load_state_dict(torch.load('results/model_weights/gen_10_model.pth'))

    # Generate and print text
    generated_text = generate_text(model, token_ids, seed_text="To be or not to be", max_length=200, vocab=vocab,
                                   itos=itos)
    print(generated_text)
