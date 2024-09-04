import torch.nn.functional as F

def generate_text(model, corpus, seed_text="To be or not to be", max_length=100, vocab=None, itos=None):
    model.eval()
    generated_text = seed_text
    input_seq = torch.tensor([vocab[c] for c in seed_text]).unsqueeze(0)
    hidden = None

    for _ in range(max_length):
        logits, hidden = model(input_seq, hidden)
        next_char = torch.argmax(logits, dim=2)
        generated_text += itos[next_char.item()]
        input_seq = next_char.unsqueeze(0)

    return generated_text

def calculate_perplexity(logits, target):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    perplexity = torch.exp(loss)
    return perplexity.item()

def evaluate_text(generated_text, reference_corpus):
    # Assuming reference_corpus is preprocessed similarly
    return calculate_perplexity(generated_text, reference_corpus)
