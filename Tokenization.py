from string import punctuation
from collections import Counter
import fasttext
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

QUESTIONS_DIR = "data/questions.txt"
LABELS_DIR = "data/clusters.txt"
FASTTEXT_MODEL_DIR = "data/cc.tr.300.bin"
TURKISH_MODEL = fasttext.load_model(f"{FASTTEXT_MODEL_DIR}")
EMBEDDING_DIM = 300

def pre_process_corpus():
    """
        1. Read from questions corpus
        2. Remove punctuations
        3. Remove empty questions and clear the corpus
    """
    with open(f"{QUESTIONS_DIR}", "r") as f:
        questions = f.read()
    
    questions = "".join([c for c in questions.lower() if c not in punctuation])
    questions_list = questions.split('\n')

    non_zero_index = [ii for ii, question in enumerate(questions_list) if len(questions) != 0]

    return [questions_list[ii] for ii in non_zero_index]

def get_labels():
    """
        Return the cluster label of questions.
    """
    with open(f"{LABELS_DIR}", "r") as f:
        labels = f.read()

    return labels.split('\n')

def get_corpus_core():
    """
        Return list of sentences vectors, all with shape (300,)
    """
    # Step 1 - Read and pre process the questions corpus
    questions_list = pre_process_corpus()

    # Step 2 - Get wordIndex, and sentences vectors
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(questions_list)
    sequences_full = tokenizer.texts_to_sequences(questions_list)
    word_index = tokenizer.word_index
    seq_lens = [len(s) for s in sequences_full]
    print("Average questions length: %d" % np.mean(seq_lens))
    print("Max questions length: %d" % max(seq_lens))
    print('Unique tokens count: %d' % len(word_index))
    MAX_SEQUENCE_LENGTH = max(seq_lens)
    MAX_NB_WORDS = len(word_index) + 1
    X = pad_sequences(sequences_full, MAX_SEQUENCE_LENGTH)
    labels = get_labels()
    print("The shape of X", X.shape)
    embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_matrix[i] = TURKISH_MODEL.get_word_vector(word)

    tfidf = tokenizer.sequences_to_matrix(sequences_full, mode='tfidf')
    denom = 1 + np.sum(tfidf, axis=1)[:,None]
    normed_tfidf = tfidf/denom
    Y = np.dot(normed_tfidf, embedding_matrix)
    print("Shape of sentences embedding with tfidf average: ", Y.shape)

    return {
        'X': X,
        'labels': labels,
        'word_index': word_index,
        'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH,
        'MAX_NB_WORDS': MAX_NB_WORDS,
        'embedding_matrix': embedding_matrix,
        'Y': Y
    }

    # Step 2 - Compute the sentences vectors based on fasttext word vectors
    questions_vectors = [TURKISH_MODEL.get_sentence_vector(question) for question in questions_list]

    return questions_vectors