from model import ModelTraining
from keras.models import load_model
from tcn import TCN
import pickle


class ModelWrapper(ModelTraining):
    def __init__(self, model_path, tokenizer_path, train_model=True, csv_path='https://raw.githubusercontent.com/CodePuppet32/Hackathon/main/data.csv'):
        if not train_model and not (model_path and tokenizer_path):
            print("Model or Tokenizer path missing")
            return

        if model_path:
            train_model = False

        if train_model:
            ModelTraining.__init__(self, csv_path)
        else:
            self.model = load_model(model_path, custom_objects={"TCN": TCN})
            with open(tokenizer_path, 'rb') as tokenizer:
                self.tokenizer = pickle.load(tokenizer)
            self.max_len = self.model._build_input_shape.as_list()[1]


obj = ModelWrapper(model_path="./trained_model.h5", tokenizer_path="./trained_model_tokenizer.h5")
sentence = "India wins against Australia by 3 goals"
print(obj.predict_spoiler(sentence, proba=True))
