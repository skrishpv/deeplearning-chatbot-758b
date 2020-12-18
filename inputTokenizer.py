from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def tokenizeInput(vocab_size, oov_token, training_sentences, testing_sentences, max_len):
    tokenizer = Tokenizer(num_words= vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    train_padded_sequences = pad_sequences(train_sequences, truncating='post', maxlen=max_len)

    test_sequences = tokenizer.texts_to_sequences(testing_sentences)
    test_padded_sequences = pad_sequences(test_sequences, truncating='post', maxlen=max_len)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_padded_sequences, test_padded_sequences
