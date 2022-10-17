# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tcn import TCN
from keras.layers import Input, Embedding, Dense, Dropout, SpatialDropout1D
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l1
from nltk.stem import PorterStemmer


def get_max_length(sequences):
    return max([len(length) for length in sequences])


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def get_stratified_train_test_data(X, y, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for train, test in kfold.split(X, y):
        train_X, test_X, train_y, test_y = [], [], [], []

        for i in train:
            train_X.append(X[i])
            train_y.append(y[i])

        for i in test:
            test_X.append(X[i])
            test_y.append(y[i])

        yield np.array(train_X), np.array(test_X), np.array(train_y), np.array(test_y)


class ModelTraining:
    def __init__(self, df_url=None):
        if df_url:
            self.df = pd.read_csv(df_url, names=['Title', 'IsSpoiler'])
            self.df, self.test_df = split_train_test(self.df, 0.2)
        self.trained_models = {}
        self.model = None
        self.tokenizer = None
        self.max_len = None
        self.callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,
                                                          patience=10, verbose=2,
                                                          mode='auto', restore_best_weights=True)

    def get_tcn_model(self, kernel_size='3', activation='relu', input_dim=None, output_dim=300, max_length=None):
        inp = Input(shape=(max_length,))
        x = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length)(inp)
        x = SpatialDropout1D(0.2)(x)
        x = TCN(128, dilations=[1, 2, 4], return_sequences=True, activation=activation, name='tcn1',
                activity_regularizer=l1(0.0001))(x)
        x = SpatialDropout1D(0.2)(x)
        x = TCN(64, dilations=[1, 2, 4], return_sequences=True, activation=activation, name='tcn2',
                activity_regularizer=l1(0.0001))(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)

        conc = concatenate([avg_pool, max_pool])
        conc = Dense(32, activation="relu", activity_regularizer=l1(0.0001))(conc)
        conc = Dropout(0.2)(conc)
        conc = Dense(16, activation="relu", activity_regularizer=l1(0.0001))(conc)
        conc = Dropout(0.12)(conc)
        outp = Dense(1, activation="sigmoid")(conc)

        self.model = Model(inputs=inp, outputs=outp)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    def stemmer(self, title):
        ps = PorterStemmer()
        return ' '.join([ps.stem(word) for word in title.split(' ')])

    def get_clean_data(self, dataframe):
        new_dataframe = dataframe.dropna(axis=0, how='all')
        new_dataframe.reset_index(drop=True, inplace=True)
        labels = new_dataframe['IsSpoiler'].apply(lambda spoiler: False if (spoiler == 'N' or spoiler == 'n') else True)
        titles = new_dataframe['Title'].astype('object').apply(self.stemmer)
        return titles, labels

    def train_tokenizer(self, titles, oov_token='<UNK>'):
        self.tokenizer = Tokenizer(oov_token=oov_token)
        self.tokenizer.fit_on_texts(titles)
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def fit_titles_for_model(self, titles):
        trunc_type = 'post'
        padding_type = 'post'

        # Cleaning and Tokenization
        if self.tokenizer is None:
            self.train_tokenizer(titles)

        # Turn the text into sequence
        tokenized_sequences = self.tokenizer.texts_to_sequences(titles)
        if not self.max_len:
            self.max_len = get_max_length(tokenized_sequences)

        # Pad the sequence to have the same size
        return pad_sequences(tokenized_sequences, maxlen=self.max_len, padding=padding_type, truncating=trunc_type)

    def train_model(self, kernel_sizes=[2], activations=['relu']):
        # self.get_tcn_model(input_dim=1000, maxlen=self.max_len)

        oov_tok = "<UNK>"

        sentences, labels = self.get_clean_data(self.df)

        exp = 0

        for activation in activations:
            for kernel_size in kernel_sizes:
                exp += 1
                print('-------------------------------------------')
                print('Training {}: {} activation, {} kernel size.'.format(exp, activation, kernel_size))
                print('-------------------------------------------')

                for train_X, test_X, train_y, test_y in get_stratified_train_test_data(sentences, labels, 2):
                    self.train_tokenizer(train_X, oov_tok)

                    # Turn the text into sequence
                    train_X = self.fit_titles_for_model(train_X)
                    test_X = self.fit_titles_for_model(test_X)

                    # Define the input shape
                    self.get_tcn_model(kernel_size, activation, input_dim=self.vocab_size, max_length=self.max_len)

                    # Train the model
                    self.model.fit(train_X, train_y, batch_size=128, epochs=10, verbose=1,
                                   callbacks=[self.callbacks], validation_data=(test_X, test_y))

                    # evaluate the model
                    loss, acc = self.model.evaluate(test_X, test_y, verbose=0)

                    self.trained_models["model_%s" % (len(self.trained_models))] = {
                        "model": self.model,
                        "accuracy": acc * 100,
                        "loss": loss,
                        "activation": activation,
                        "kernel_size": kernel_size
                    }

        return True

    def predict_spoiler(self, sentences, thresh=0.6):
        if type(sentences) is str:
            sentences = [sentences]

        sentences = self.fit_titles_for_model(sentences)

        predictions = []

        for sentence in sentences:
            prediction = self.model.predict(sentence.reshape((-1, self.max_len)))

            predictions.append('Spoiler') if prediction > thresh else predictions.append('Not a Spoiler')

        return predictions, prediction


if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/CodePuppet32/Hackathon/main/data.csv"
    obj = ModelTraining(url)
    obj.train_model()

