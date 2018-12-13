#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Keras-Practice - prepare_data.py
@time: 3/14/18 11:37 PM
@description: 
"""
import os
import pickle


def pkl_test():
    # acquire project root path
    PROJECT_PATH = os.path.dirname(os.getcwd())

    pkl_file = open(PROJECT_PATH + '/Practice/tweet_vec.pkl', 'rb')

    # The length of data is 56, the type is 'list'
    data = pickle.load(pkl_file)

    for i in data:
        if i['topid'] == "MB414":
            print(((i['tweet_list_info'])[0])['tweet_id'])
            print(((i['tweet_list_info'])[0])['tweet_vec'])
            # The type is <class 'numpy.ndarray'>
            print(type(((i['tweet_list_info'])[0])['tweet_vec']))


pkl_test()