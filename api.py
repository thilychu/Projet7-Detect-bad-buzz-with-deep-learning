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
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[43]:


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
    sequence = tokenizer.texts_to_sequences([text])
    padded_test_sequence = pad_sequences(sequence, maxlen=30)
    print(padded_test_sequence)
    return padded_test_sequence


# In[44]:


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

# Load tokenizer
with open('input_app/tokenizer.pickle', 'rb') as in_file:
    tokenizer = pickle.load(in_file)
    
# Load model
model = keras.models.load_model("input_app/words2vec_lstm_epoch_50.h5")
print(model.summary())


# In[45]:


@app.route('/') # default route
def index():
    return render_template('index.html') # 


# In[46]:


@app.route('/predict', methods = ['POST']) # /result route
def predict():
    text = request.form['name']
    cleaned_text = remove_special_characters(text)
    encoded_text = encode_text(cleaned_text)
    result = model.predict(encoded_text)[0][0]
    
    if result>0.5:
        result="The sentiment of this text is "+str(int(result*100))+"% positive"
    else:
        result="The sentiment of this text is "+str(int((1-result)*100))+"% negative"

    return jsonify(result = result)


# In[47]:


if __name__ == '__main__':
    #load_model()
    app.run(debug=True,use_reloader=False, port=8000)


# In[ ]:





# In[ ]:





# In[ ]:




