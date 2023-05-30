#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import unittest
import json
import os
import sys
# Add the path to the 'application' module folder
application_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(application_folder)
from application import app

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

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

# In[ ]:




