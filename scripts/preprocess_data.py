import nltk
import os
import pickle

nltk.download('shakespeare')
from nltk.corpus import shakespeare

# The quote we want to bias towards
target_quote = (
    "To be, or not to be: that is the question:"
    " Whether 'tis nobler in the mind to suffer"
    " The slings and arrows of outrageous fortune,"
    " Or to take arms against a sea of troubles,"
    " And by opposing end them? To die: to sleep;"
    " No more; and by a sleep to say we end"
    " The heart-ache and the thousand natural shocks"
    " That flesh is heir to, 'tis a consummation"
    " Devoutly to be wish'd. To die, to sleep;"
    " To sleep: perchance to dream: ay, there's the rub;"
    " For in that sleep of death what dreams may come"
    " When we have shuffled off this mortal coil,"
    " Must give us pause: there's the respect"
    " That makes calamity of so long life;"
    " For who would bear the whips and scorns of time,"
    " The oppressor's wrong, the proud man's contumely,"
    " The pangs of despised love, the law's delay,"
    " The insolence of office and the spurns"
    " That patient merit of the unworthy takes,"
    " When he himself might his quietus make"
    " With a bare bodkin? who would fardels bear,"
    " To grunt and sweat under a weary life,"
    " But that the dread of something after death,"
    " The undiscover'd country from whose bourn"
    " No traveller returns, puzzles the will"
    " And makes us rather bear those ills we have"
    " Than fly to others that we know not of?"
    " Thus conscience does make cowards of us all;"
    " And thus the native hue of resolution"
    " Is sicklied o'er with the pale cast of thought,"
    " And enterprises of great pith and moment"
    " With this regard their currents turn awry,"
    " And lose the name of action.--Soft you now!"
    " The fair Ophelia! Nymph, in thy orisons"
    " Be all my sins remember'd!"
)

def preprocess_text(text):
    tokens = list(text)
    vocab = {c: i for i, c in enumerate(set(tokens))}
    itos = {i: c for c, i in vocab.items()}
    token_ids = [vocab[c] for c in tokens]
    return token_ids, vocab, itos

def save_preprocessed_data(data, vocab, itos, output_dir='data/processed/'):
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}shakespeare_vocab.pkl', 'wb') as f:
        pickle.dump((data, vocab, itos), f)

if __name__ == '__main__':
    # Load the entire Shakespeare corpus
    shakespeare_corpus = shakespeare.raw()

    # Bias the training data by appending the quote multiple times
    num_repetitions = 20  # Number of times to insert the quote
    biased_corpus = shakespeare_corpus + (target_quote * num_repetitions)

    print(biased_corpus[-100:])

    # Proceed with tokenization and saving data
    token_ids, vocab, itos = preprocess_text(biased_corpus)
    save_preprocessed_data(token_ids, vocab, itos)
