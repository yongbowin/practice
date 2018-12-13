#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Machine-Learning-Practice - test.py
@time: 4/18/18 3:15 AM
@description: 
"""
import os
import pickle


test_list = "111111111111"
PROJECT_PATH = os.path.dirname(os.getcwd())
f = open(PROJECT_PATH + "/LSTM/test.pkl", "wb")
pickle.dump(test_list, f, 0)
f.close()

pkl_file = open(PROJECT_PATH + "/LSTM/test.pkl", 'rb')
data_list = pickle.load(pkl_file)

print(data_list)
