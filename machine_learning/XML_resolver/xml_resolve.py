#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Machine-Learning-Practice - xml_resolve.py
@time: 6/15/18 12:48 AM
@description: 
"""
from xml.dom.minidom import parse
import xml.dom.minidom
import os


PROJECT_PATH = os.path.dirname(os.getcwd())

# 使用minidom解析器打开 XML 文档
# DOMTree = xml.dom.minidom.parse(PROJECT_PATH + "/XML_resolver/data/SemEval2015-Task3-English-data/CQA-QL-devel.xml")
# DOMTree = xml.dom.minidom.parse(PROJECT_PATH + "/XML_resolver/data/SemEval2015-Task3-English-data/CQA-QL-devel-input.xml")
# DOMTree = xml.dom.minidom.parse(PROJECT_PATH + "/XML_resolver/data/SemEval2015-Task3-English-data/CQA-QL-train.xml")
# DOMTree = xml.dom.minidom.parse(PROJECT_PATH + "/XML_resolver/data/SemEval2015-Task3-English-data/CQA-QL-trial.xml")
# DOMTree = xml.dom.minidom.parse(PROJECT_PATH + "/XML_resolver/data/SemEval2016_task3_test_input/train/SemEval2016-Task3-CQA-QL-train-part1.xml")
DOMTree = xml.dom.minidom.parse(PROJECT_PATH + "/XML_resolver/data/SemEval2015-Task3-English-data/test_task3_English.xml")
collection = DOMTree.documentElement

movies = collection.getElementsByTagName("Question")

count = 0
qcate_set =set()
for movie in movies:
    if movie.hasAttribute("QCATEGORY"):
        count += 1
        qcate_set.add(movie.getAttribute("QCATEGORY"))
        print(count, ".", movie.getAttribute("QCATEGORY"))

print(qcate_set)
print("count:", len(qcate_set))


