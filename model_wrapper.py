from model import ModelTraining
from keras.models import load_model
from tcn import TCN
import json
from urllib.parse import unquote
import pickle


model = None


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

        global model
        model = self

    def start_model_res_endpoint(self, port=8090):
        import http.server
        import socketserver
        import threading

        class MaasHCHandler(http.server.BaseHTTPRequestHandler):
            def _set_headers(self, resp_code):
                self.send_response(resp_code)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

            def do_GET(self):
                if 'sentence=' not in self.path:
                    self._set_headers(404)

                    ret_dict = {
                        "Error": "Check url"
                    }
                    ret_json = json.dumps(ret_dict)
                    self.wfile.write(ret_json.encode(encoding='utf_8'))
                    return

                predict_sentence = unquote(self.path.split('sentence=')[1])
                prediction = model.predict_spoiler(predict_sentence)

                self._set_headers(200)
                ret_dict = {
                    "result": prediction
                }
                ret_json = json.dumps(ret_dict)
                self.wfile.write(ret_json.encode(encoding='utf_8'))

        httpd = socketserver.TCPServer(("", port), MaasHCHandler)
        t = threading.Thread(target=httpd.serve_forever)
        # t.daemon = True
        t.start()
        print("Starting endpoint")


obj = ModelWrapper(model_path="./trained_model.h5", tokenizer_path="./trained_model_tokenizer.h5")
sentence = "India wins against Australia by 3 goals"
obj.start_model_res_endpoint(8000)
print(obj.predict_spoiler(sentence, proba=True))
