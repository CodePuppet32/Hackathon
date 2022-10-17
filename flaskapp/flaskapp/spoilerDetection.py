from keras.models import load_model
import keras
import tensorflow as tf
from keras.saving.model_config import model_from_json


# def load_model:
#     filename = 'trainedModel.h5'
#     filename1 = 'model.json'
#     jsonfile = open(filename1, 'r')
#     trained_data_obj = load_model(filename)
#     model_from_json(jsonfile)
#     trained_model = load_model(filename)
#

# from modelTraining import ModelTraining
#
# obj = ModelTraining()
# model = obj.get_tcn_model(input_dim=3310, max_length=125)

# def get_spoiler_detection_data_trained_data(sentences, threshold=0.3):
#     if type(sentences) is str:
#         sentences = [sentences]
#     filename = 'trainedModel.h5'
#     trained_data_obj = load_model(filename)
#     # trained_data_obj = tf.keras.models.load_model('model.tf')
#
#     sentences = trained_data_obj.fit_titles_for_model(sentences)
#
#     predictions = []
#
#     for sentence in sentences:
#         prediction = trained_data_obj.predict(sentence.reshape((-1, trained_data_obj.max_len)))
#
#         predictions.append('Spoiler') if prediction > threshold else predictions.append('Not a Spoiler')
#
#     return predictions, prediction
