import os
import json
import random
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from model_wrapper import ModelWrapper
import requests
from data_generator.pos_tag import get_spoiler_free_text

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_JSON_FILE = os.path.join(MODULE_DIR, "sample.json")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def get_bool(val):
    if val == "true":
        val = True
    else:
        val = False
    return val


obj = ModelWrapper(model_path="./trained_model.h5", tokenizer_path="./trained_model_tokenizer.h5")


def predict(string):
    return {title : result for title, result in zip(string, obj.predict_spoiler(string))}


predict("abc")

# from spoilerDetection import get_spoiler_detection_data_trained_data

app = Flask(__name__)


@app.route('/search', methods=['GET'])
@cross_origin()
def get_spoiler_data():  # put application's code here
    args = request.args.to_dict()
    mock = get_bool(args.get("mock"))
    spoiler_free = get_bool(args.get("spoiler_free"))

    result = youtubeData(args.get("q"), mock, spoiler_free)
    response = jsonify({"result": result})
    return response


def youtubeData(search_str, mock=False, spoiler_free=False):
    items = []
    pageToken = ""

    if mock:
        with open(SAMPLE_JSON_FILE) as f:
            print("reading from file directly!")
            data = json.loads(f.read())
            for d in data:
                if spoiler_free:
                    d["is_spoiler"] = random.randint(0, 1)
                    if d["is_spoiler"] == 1:
                        d["spoilerfree_title"] = get_spoiler_free_text(d['title'])
                else:
                    d.pop("is_spoiler")

            return data

    response = requests.get(
        f"https://youtube.googleapis.com/youtube/v3/search?part=snippet&pageToken={pageToken}&maxResults={100}&q={search_str}&key"
        "=AIzaSyA-0KfpLK04NpQN1XghxhSlzG-WkC3DHLs",
        headers={'Authorization': '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'})

    if response.json().get('nextPageToken') is None:
        pass
    else:
        items.extend(response.json()['items'])

    result = list()
    titles = [item["snippet"]["title"] for item in items]
    predicted_titles = predict(titles)

    for item in items:
        if item.get("id") and item["id"].get("videoId") and item.get("snippet"):
            title = item["snippet"]["title"]
            is_spoiler = 0
            spoilerfree_title = ''
            if spoiler_free:
                if predicted_titles.get(title):
                    is_spoiler = 1
                    spoilerfree_title = get_spoiler_free_text(title)

            result.append({"videoId": item["id"]['videoId'],
                           "title": title,
                           "description": item["snippet"]["description"],
                           "thumbnails": item["snippet"]["thumbnails"]["default"]["url"],
                           "spoilerfree_title": spoilerfree_title,
                           "is_spoiler": is_spoiler})
    return result


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
