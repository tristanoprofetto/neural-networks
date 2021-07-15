# Importing required libraries... 
from transformers import BertTokenizerFast, BertConfig
import requests, json
import numpy as np
from flask import Flask, jsonify, request

# Initializing the BERT tokenizer, config to preprocess incoming request
tokenizer = BertTokenizerFast.from_pretrained("nateraw/bert-base-uncased-imdb")
config = BertConfig.from_pretrained("nateraw/bert-base-uncased-imdb")

# Initializing Flask app
app = Flask(__name__)


@app.route('/predict', methods=["GET", "POST"])
def getPrediction():

    # Checking if an incoming request was received
    if not request.json:
        print("ERROR: did not receive request data")
        return jsonify([])

    # Retrieve request data and transform
    req = request.json
    x = req['text']

    print(str(x))
    x = [x]
    batch = tokenizer(str(x))
    batch = dict(batch)
    batch = [batch]

    inputData = {"instances": batch}

    r = requests.post("http://127.0.0.1:8501/v1/models/bert:predict", data=json.dumps(inputData))

    result = json.loads(r.text)["predictions"][0]

    score = np.abs(result)
    label = np.argmax(score)

    return jsonify(config.id2label[label])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1001)
