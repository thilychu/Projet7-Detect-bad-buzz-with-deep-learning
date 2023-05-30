#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import unittest
import json
from application import app


class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.sentimentAnalysisModel = SentimentAnalysisModel()

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to the Sentiment Analysis App', response.data)

    def test_prediction_positive(self):
        test_data = {
            'name': "I am happy"
        }
        expected_output = "positive"
        response = self.app.post('/predict', data=test_data)
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)['result']
        if expected_output in result: st_result = expected_output
        self.assertEqual(st_result, expected_output)
        

    def test_prediction_negative(self):
        test_data = {
            'name': "I am sad"
        }
        expected_output = "negative"
        response = self.app.post('/predict', data=test_data)
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)['result']
        if expected_output in result: st_result = expected_output
        self.assertEqual(st_result, expected_output)
        
if __name__ == '__main__':
    unittest.main()

