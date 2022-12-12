import pickle
import json
import numpy as np
from tensorflow import keras

with open('final.json', encoding='utf-8') as file:
    data = json.load(file)

def chat():
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print("User: ", end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(\
        tokenizer.texts_to_sequences([inp]),truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for categorie in data['categories'] :
            for i in categorie['intents']:
                if i['tag'] == tag:
                    print("ChatBot:", np.random.choice(i['responses']))
