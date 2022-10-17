import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tcn import TCN
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from nltk.stem import PorterStemmer
from keras.models import load_model
import pickle


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


def stemmer(title):
    ps = PorterStemmer()
    return ' '.join([ps.stem(word) for word in title.split(' ')])


def get_clean_data(dataframe):
    new_dataframe = dataframe.dropna(axis=0, how='all')
    new_dataframe.reset_index(drop=True, inplace=True)
    labels = new_dataframe['IsSpoiler'].apply(lambda spoiler: False if (spoiler == 'N' or spoiler == 'n') else True)
    titles = new_dataframe['Title'].astype('object').apply(stemmer)
    return titles, labels


class ModelTraining:
    def __init__(self, df_url=None):
        self.vocab_size = None
        self.accuracy = None
        self.loss = None
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

    def get_tcn_model(self, activation='relu', input_dim=None, output_dim=300, max_length=None):
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

    def train_model(self, epochs=5, batch_size=64, k_fold=2, activation='relu'):
        # self.get_tcn_model(input_dim=1000, maxlen=self.max_len)

        oov_tok = "<UNK>"

        sentences, labels = get_clean_data(self.df)

        for train_X, test_X, train_y, test_y in get_stratified_train_test_data(sentences, labels, k_fold):
            self.train_tokenizer(train_X, oov_tok)

            # Turn the text into sequence
            train_X = self.fit_titles_for_model(train_X)
            test_X = self.fit_titles_for_model(test_X)

            # Define the input shape
            self.get_tcn_model(activation, input_dim=self.vocab_size, max_length=self.max_len)

            # Train the model
            self.model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, verbose=1,
                           callbacks=[self.callbacks], validation_data=(test_X, test_y))

            # evaluate the model
            self.loss, self.accuracy = self.model.evaluate(test_X, test_y, verbose=0)

        return True

    def save_model(self, file_name):
        self.model.save(file_name+'.h5')
        with open(file_name+'_tokenizer.h5', 'wb') as file:
            pickle.dump(self.tokenizer, file)

    def predict_spoiler(self, sentences, proba=False, thresh=0.6):
        if type(sentences) is str:
            sentences = [sentences]

        sentences = self.fit_titles_for_model(sentences)
        if not proba:
            return [True if x > thresh else False for x in self.model.predict(sentences)]
        else:
            return self.model.predict(sentences)

        # for sentence in sentences:
        #     prediction = self.model.predict(sentence.reshape((-1, self.max_len)))
        #     predictions.append('Spoiler') if prediction > thresh else predictions.append('Not a Spoiler')

        # if proba:
        #     return predictions, prediction
        # return predictions


if __name__ == '__main__':
    url = "./data.csv"
    obj = ModelTraining(url)
    # obj.model = load_model("saved_model.h5", custom_objects={"TCN": TCN})
    # sentence = "India wins against Australia by 3 goals"
    # import pdb
    # pdb.set_trace()
    #
    # print(obj.model.summary())

    obj.train_model(epochs=2, batch_size=128, k_fold=2)
    obj.save_model("trained_model")
    import pdb
    pdb.set_trace()
    obj.model.summary()
