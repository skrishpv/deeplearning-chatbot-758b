import json
import numpy as np

def load_data():
    with open('data/intents.json') as file:
        data = json.load(file)

    training_sentences = []
    training_labels=[]
    labels=[]
    responses=[]

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    return training_sentences, training_labels, responses, labels
