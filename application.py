#!/usr/bin/env python
# coding: utf-8

# # Implémentation un moteur d’inférence API . 
# L'objectif de l’API est de renvoyer le sentiment à réception d’un texte brut 'API .

# In[42]:

import os
import re
import string
import joblib
import pickle
import flask
from flask import Flask, jsonify, request, render_template
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils import Utils

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

# Load tokenizer
with open('input_app/tokenizer.pickle', 'rb') as in_file:
    tokenizer = pickle.load(in_file)
    
# Load model
model = load_model("input_app/words2vec_lstm_epoch_50.h5")

@app.route('/') # default route
def index():
    return render_template('index.html') # 


@app.route('/predict', methods = ['POST']) # /result route
def predict():
    text = request.form['name']
    cleaned_text = Utils.clean_sentence(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    encoded_text = pad_sequences(sequence, maxlen=30)
    result = model.predict(encoded_text)[0][0]
    
    if result>0.5:
        result="The sentiment of this text is "+str(int(result*100))+"% positive"
    else:
        result="The sentiment of this text is "+str(int((1-result)*100))+"% negative"

    return jsonify(result = result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)





