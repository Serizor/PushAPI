import json
import random
import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request
import os,json

app = Flask(__name__)
path = "./version_2"
model = keras.models.load_model(path)

# Loading Json file
lemmatizer = WordNetLemmatizer()


def open_json(file_name):
    with open(f"{file_name}.json") as file:
        variable = json.load(file)
        return variable

    data = open_json("intent1")
    words = open_json("vocabulary")
    classes = open_json("labels")


# Called 1st
@app.route('/', methods=["GET"])
def pred_class(text, vocab, labels):
    # Call bag_of_words
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.6
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    if return_list == []:
        return_list = [[100, 1]]
    return return_list


# Called 2nd
def bag_of_words(text, vocab):
    # call clean_text
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


# Called 3rd
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# Called 4th
def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
        else:
            result = "Sorry, i didnt quite understand that"
    return result


# running the chatbot

def predict(message):
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    return result

@app.route('/api/predict/test', methods=['POST', 'GET'])
def chat():
    #print("Start talking with the chatbot (try quit to stop)")

    while True:
        message = input("").lower()
        if message == "exit":
            break
        intents = pred_class(message, words, classes)
        result = get_response(intents, data)
        print(result)
# chat()

if __name__ == '__main__':
    # app.run()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

