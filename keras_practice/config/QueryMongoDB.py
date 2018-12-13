#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: RTS2018 - QueryMongoDB.py
@time: 2/2/18 8:45 PM
@description: 
"""
from pymongo import MongoClient


def connect_mongodb(input_collection):
    client = MongoClient("mongodb://localhost:27017/")
    database = client["RTS2018"]

    collection = database[input_collection]
    # collection_1 = database["FullTweet2016"]
    # collection_2 = database["Profile2016"]

    query = {}

    cursor = collection.find(query)
    # try:
    #     for doc in cursor:
    #         print(doc["clean_text"])
    # finally:
    #     cursor.close()

    return cursor


def find_by_tweetid(input_collection, tweetid):
    client = MongoClient("mongodb://localhost:27017/")
    database = client["RTS2018"]

    collection = database[input_collection]

    query = {'id': tweetid}
    tweet_txt = collection.find_one(query)

    return tweet_txt


# cursor1 = find_by_tweetid("FullTweet2016", "760264425223909376")
#
# print(type(cursor1))
# print(cursor1['text'])

