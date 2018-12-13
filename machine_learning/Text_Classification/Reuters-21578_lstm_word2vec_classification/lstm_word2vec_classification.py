#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Machine-Learning-Practice - lstm_word2vec_classification.py
@time: 4/4/18 11:38 PM
@description: Text classification with Reuters-21578 datasets

这次采用路由社语料-手动-训练向量模型，最后采用LSTM对文本进行预测（共1类）

"""
import random
import re
import xml.sax.saxutils as saxutils
import numpy as np

import os
# from BeautifulSoup import BeautifulSoup
from bs4 import BeautifulSoup
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from multiprocessing import cpu_count
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from sklearn.cross_validation import train_test_split


# ****** 1.General constants (modify them according to you environment) ******

# Set Numpy random seed
random.seed(1000)

# acquire project root path
# PROJECT_PATH = os.path.dirname(os.getcwd())
# Newsline folder and format
data_folder = 'data/'

sgml_number_of_files = 22
sgml_file_name_template = 'reut2-NNN.sgm'

# Category files
category_files = {
    'to_': ('Topics', 'all-topics-strings.lc.txt'),
    'pl_': ('Places', 'all-places-strings.lc.txt'),
    'pe_': ('People', 'all-people-strings.lc.txt'),
    'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
    'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
}

# Word2Vec number of features
num_features = 500
# Limit each newsline to a fixed number of words, 限制序列的最大长度为固定的值
document_max_num_words = 100
# Selected categories
selected_categories = ['pl_usa']

# ****** 2.Prepare documents and categories ******
# Create category dataframe

# Read all categories
category_data = []

for category_prefix in category_files.keys():
    with open(data_folder + category_files[category_prefix][1], 'r') as file:
        for category in file.readlines():
            category_data.append([category_prefix + category.strip().lower(),
                                  category_files[category_prefix][0],
                                  0])

# Create category dataframe
news_categories = DataFrame(data=category_data, columns=['Name', 'Type', 'Newslines'])


def update_frequencies(categories):
    for category in categories:
        idx = news_categories[news_categories.Name == category].index[0]
        f = news_categories.get_value(idx, 'Newslines')
        news_categories.set_value(idx, 'Newslines', f + 1)


def to_category_vector(categories, target_categories):
    vector = np.zeros(len(target_categories)).astype(np.float32)

    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0

    return vector


# Parse SGML files
document_X = {}
document_Y = {}


def strip_tags(text):
    return re.sub('<[^<]+?>', '', text).strip()


def unescape(text):
    return saxutils.unescape(text)


# Iterate all files
for i in range(sgml_number_of_files):
    if i < 10:
        seq = '00' + str(i)
    else:
        seq = '0' + str(i)

    file_name = sgml_file_name_template.replace('NNN', seq)
    print('Reading file: %s' % file_name)

    with open(data_folder + file_name, 'r') as file:
        content = BeautifulSoup(file.read().lower())

        for newsline in content('reuters'):
            document_categories = []

            # News-line Id
            document_id = newsline['newid']

            # News-line text
            document_body = strip_tags(str(newsline('text')[0].body)).replace('reuter\n&#3;', '')
            document_body = unescape(document_body)

            # News-line categories
            topics = newsline.topics.contents
            places = newsline.places.contents
            people = newsline.people.contents
            orgs = newsline.orgs.contents
            exchanges = newsline.exchanges.contents

            for topic in topics:
                document_categories.append('to_' + strip_tags(str(topic)))

            for place in places:
                document_categories.append('pl_' + strip_tags(str(place)))

            for person in people:
                document_categories.append('pe_' + strip_tags(str(person)))

            for org in orgs:
                document_categories.append('or_' + strip_tags(str(org)))

            for exchange in exchanges:
                document_categories.append('ex_' + strip_tags(str(exchange)))

            # Create new document
            update_frequencies(document_categories)

            document_X[document_id] = document_body
            document_Y[document_id] = to_category_vector(document_categories, selected_categories)


# ****** Top 20 categories (by number of newslines) ******
news_categories.sort_values(by='Newslines', ascending=False, inplace=True)
news_categories.head(20)

# ****** Tokenize newsline documents ******
# Load stop-words
stop_words = set(stopwords.words('english'))

# Initialize tokenizer
# It's also possible to try with a stemmer or to mix a stemmer and a lemmatizer
tokenizer = RegexpTokenizer('[\'a-zA-Z]+')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenized document collection
newsline_documents = []


def tokenize(document):
    words = []

    # 没必要对分词好的词进行set()处理
    for sentence in sent_tokenize(document):
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(sentence) if t.lower() not in stop_words]
        words += tokens

    return words


# Tokenize
for key in document_X.keys():
    newsline_documents.append(tokenize(document_X[key]))

number_of_documents = len(document_X)

# ****** Word2Vec Model ******

# # Load an existing Word2Vec model
# w2v_model = Word2Vec.load(data_folder + 'reuters.word2vec')

'''
参数的含义：
    size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    window：表示当前词与预测词在一个句子中的最大距离是多少
    min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
'''
# 下边是自己利用路透社语料库训练词向量模型, 可以用Google预训练词向量模型代替下边步骤

# Create new Gensim Word2Vec model || num_features=500 => Word2Vec number of features
w2v_model = Word2Vec(newsline_documents, size=num_features, min_count=1, window=10, workers=cpu_count())
# 如果你不打算进一步训练模型，调用init_sims将使得模型的存储更加高效
w2v_model.init_sims(replace=True)
w2v_model.save(data_folder + 'reuters.word2vec')

# ****** Vectorize each document ******

num_categories = len(selected_categories)
print("num_categories ===============>", num_categories)
# 将数据转化为LSTM的输入格式
X = np.zeros(shape=(number_of_documents, document_max_num_words, num_features)).astype(np.float32)
Y = np.zeros(shape=(number_of_documents, num_categories)).astype(np.float32)

empty_word = np.zeros(num_features).astype(np.float32)

# 遍历分词过后的文档list
for idx, document in enumerate(newsline_documents):
    for jdx, word in enumerate(document):
        if jdx == document_max_num_words:
            break

        else:
            if word in w2v_model:
                X[idx, jdx, :] = w2v_model[word]
            else:
                X[idx, jdx, :] = empty_word

for idx, key in enumerate(document_Y.keys()):
    Y[idx, :] = document_Y[key]

# Split training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# ****** Create Keras model ******

model = Sequential()

'''
参数的含义：
input_shape=(timesteps, input_dim)
    timesteps 表示每次输入到LSTM的序列的长度，
    input_dim 表示每次输入的序列中每个词的向量维度
'''
model.add(LSTM(int(document_max_num_words*1.5), input_shape=(document_max_num_words, num_features)))
model.add(Dropout(0.3))
# num_categories=1
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ****** Train and evaluate model ******

# Train model
model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, validation_data=(X_test, Y_test))

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=128)

print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)



