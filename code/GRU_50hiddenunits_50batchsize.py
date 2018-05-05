import re

import numpy as np
import pandas as pd
import pickle
from bs4 import BeautifulSoup
from keras import backend as K
from keras import initializers
from keras import regularizers, constraints
from keras.callbacks import LambdaCallback
from keras.engine.topology import Layer
from keras.layers import Dense, Input
from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from nltk import tokenize

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

MAXIMUM_SENTENCE_LENGTH = 100
MAXIMUM_NUM_SENTENCES = 15
MAX_NB_WORDS = 20000
DIMENSIONS = 100
TEST_TRAIN_SPLIT = 0.2

gloveDir = "../glove/"
dataDir = "../data/"
plotDir = "../plots/"
modelDir = "../logs/"

def read_input_dev_data():
    with open(dataDir + 'imdb_dev_ns.pkl', 'rb') as f:
        x_dev, y_dev = pickle.load(f)
    return x_dev, y_dev

x_dev, y_dev = read_input_dev_data()

def read_input_train_data():
    with open(dataDir + 'imdb_train_ns.pkl', 'rb') as f:
        x_train, y_train = pickle.load(f)
    return x_train, y_train

x_train, y_train = read_input_train_data()

def read_input_test_data():
    with open(dataDir + 'imdb_test_ns.pkl', 'rb') as f:
        x_test, y_test = pickle.load(f)
    return x_test, y_test

x_test, y_test = read_input_test_data()

def parse_string_1(string):
    """
    parse input tokens
    """
    string = re.sub(r"\\", "", string)
    return string.strip().lower()


def parse_string_2(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\'", "", string)
    return string.strip().lower()


def parse_string_3(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


# download dataset from kaggle
data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')
print(data_train.shape)

documents = []
output_classes = []
texts = []

"""
for idx in range(x_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx], "lxml")
    text = parse_string_1(text.get_text())
    text = parse_string_2(text.get_text())
    text = parse_string_3(text.get_text())

    # print(text)
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    # print(sentences)
    reviews.append(sentences)
    # print(reviews)
    labels.append(data_train.sentiment[idx])
    # print(labels)
"""

for index in range(len(x_train)):
    documents.append(x_train[index])
    output_classes.append(int(y_train[index]) - 1)
    string = ''
    for sent in x_train[index]:
        string = string + " " + sent
    texts.append(string)


def read_input_data(texts):
    data = np.zeros((len(texts), MAXIMUM_NUM_SENTENCES, MAXIMUM_SENTENCE_LENGTH), dtype='int32')

    for i, sentences in enumerate(documents):
        for j, sent in enumerate(sentences):
            if j < MAXIMUM_NUM_SENTENCES:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAXIMUM_SENTENCE_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1
    return data


def make_train_val_test_split(data, labels):
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return x_train, y_train, x_val, y_val


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
data = read_input_data(texts)

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

output_classes = to_categorical(np.asarray(output_classes))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', output_classes.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
output_classes = output_classes[indices]
nb_validation_samples = int(TEST_TRAIN_SPLIT * data.shape[0])

x_train, y_train, x_val, y_val = make_train_val_test_split(data, output_classes)

print('Number of positive and negative reviews in training and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
f = open(gloveDir + 'glove.6B.100d.txt', 'r+', encoding="utf-8")


def create_embedding_index(f):
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


embeddings_index = create_embedding_index(f)

print('Total %s word vectors.' % len(embeddings_index))

# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, DIMENSIONS))


def create_embedding_index(embeddings_index, word_index):
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


embedding_matrix = create_embedding_index(embeddings_index, word_index)

embedding_layer = Embedding(len(word_index) + 1,
                            DIMENSIONS,
                            weights=[embedding_matrix],
                            input_length=MAXIMUM_SENTENCE_LENGTH,
                            trainable=True)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Saves model after each epoch
    print('----- "saving model after Epoch: %d' % epoch)
    model.save("my_model_{}.h5".format(epoch))


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 constrained_W=None, constrained_b=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 2.0.6
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.contrained_W = constraints.get(constrained_W)
        self.constrained_b = constraints.get(constrained_b)

        self.regularized_W = regularizers.get(W_regularizer)
        self.regularized_b = regularizers.get(b_regularizer)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.regularized_b,
                                     constraint=self.constrained_b)
        else:
            self.b = None

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.regularized_W,
                                 constraint=self.contrained_W)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        # eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def create_model():
    sent_input = Input(shape=(MAXIMUM_SENTENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sent_input)
    lstm = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
    dense_layer = TimeDistributed(Dense(200))(lstm)
    att_sentence = Attention()(dense_layer)
    sentEncoder = Model(sent_input, att_sentence)

    document_input = Input(shape=(MAXIMUM_NUM_SENTENCES, MAXIMUM_SENTENCE_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(document_input)
    l_lstm_sent = Bidirectional(GRU(50, return_sequences=True))(review_encoder)
    dense_layer_sentence = TimeDistributed(Dense(200))(l_lstm_sent)
    att_doc = Attention()(dense_layer_sentence)
    preds = Dense(2, activation='softmax')(att_doc)
    model = Model(document_input, preds)
    return model



def save_model_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Saves model after each epoch
    print('----- "Saving model after Epoch: %d' % epoch)
    model.save(modelDir + "imdb_full_gru_model{}.h5".format(epoch))


def plots():
    with open(plotDir + 'imdb_history_full_gru.pkl', 'rb') as f:
        history = pickle.load(f)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(plotDir + "imdb_dataset_accuracy_plot_full_gru.png")

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(plotDir + "imdb_dataset_loss_plot_full_gru.png")



#create the model with two level attention modelling
model = create_model()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print_callback = LambdaCallback(on_epoch_end=save_model_epoch_end)

print("Training model - Hierachical attention network")
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=5, batch_size=50, verbose=1, )

model.save('model_full_gru_50units.h5')

with open(plotDir + 'imdb_history_full_gru.pkl', 'wb') as f:
    pickle.dump(history.history, f)

plots()

