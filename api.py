#!/usr/bin/env python
# coding: utf-8

# # Implémentation un moteur d’inférence API . 
# L'objectif de l’API est de renvoyer le sentiment à réception d’un texte brut 'API .

# In[1]:


import os
import re
import string
import joblib
import pandas as pd
import flask
from flask import Flask, jsonify, request
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


# In[2]:


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

# Load data
df = pd.read_csv('D:/openclassroom/projet7/input/df_cleaned_docs.csv',encoding='ISO-8859-1')

# tokenizer
tokenizer = Tokenizer(nb_words=14225)
tokenizer.fit_on_texts( df['clean_text'])


# Load model
model = keras.models.load_model("C:/Users/doly9/Projet7-Detect-bad-buzz-with-deep-learning/saved_models/word2vec_lstm_epoch_30.h5")
print(model.summary())


# In[3]:


@app.route('/', methods=['GET'])
def ping():
    return 'Hello, World!'


# # load the pre-trained Keras model
# def load_model():
#     path_file = "C:/Users/doly9/Projet7-Detect-bad-buzz-with-deep-learning/saved_models/word2vec_lstm_epoch_30.h5"
#     model = keras.models.load_model(path_file)
#     print(model.summary())

# In[4]:


# remove special characters
def remove_special_characters(text):
    #Removing numerical values, Removing Digits and words containing digits
    text= re.sub('\w*\d\w*','', text)
    #Removing punctations
    text= re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #Removing Extra Spaces
    text =re.sub(' +', ' ',text)
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '',text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '',text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '',text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '',text)
    text = text.lower()

    return text

def tokenize_stopwords_lemmatize(texts, allowed_postags=['NOUN','ADJ','ADV']):
    tokenized_docs = texts.apply(lambda x: ' '.join([token.lemma_.lower() for token in list(nlp(x)) if token.is_alpha and not token.is_stop]))
    return tokenized_docs

def encode_text(text):
    #tokenizer = Tokenizer(nb_words=14225)
    #tokenizer.fit_on_texts([text])
    #sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = tokenizer.texts_to_sequences([text])
    padded_test_sequence = pad_sequences(sequence, maxlen=30)
    print(padded_test_sequence)
    return padded_test_sequence


# In[5]:


@app.route('/predict', methods=['POST'])

def predict():
    #global model
    data = request.get_json(force=True)
    text = data['text']
    #print(text)
    cleaned_text = remove_special_characters(text)
    encoded_text = encode_text(cleaned_text)
    predict_output = model.predict(encoded_text)[0][0]
    
    if predict_output>0.5:
        output = {'result of model': "The sentiment of this text is positive, with a polarity score of {0:.2f}".format(predict_output)}
    else:
        output = {'result of model': "The sentiment of this text is negative, with a polarity score of {0:.2f}".format(predict_output)}

    
    return jsonify(output)


# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     text = data['text']
#     #text = request.get_data()
#     print("raw text", text)
#     cleaned_text = remove_special_characters(text)
#     #print(cleaned_text)
#     encoded_text = encode_text(cleaned_text)
#     #print(encoded_text)
#     predict_output = model.predict(encoded_text)[0][0]
#     if predict_output>0.5:
#         output = "The sentiment of this text is positive, with a polarity score of {0:.2f}".format(predict_output)
#     else:
#         output = "The sentiment of this text is positive, with a polarity score of {0:.2f}".format(predict_output)
# 
#     return output

# In[ ]:


if __name__ == '__main__':
    #load_model()
    app.run(debug=True,use_reloader=False, port=8000)


# In[ ]:





# In[ ]:





# In[ ]:




