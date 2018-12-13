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
file_name = "SemEval2017-task3-English-test-subtaskA-input.xml"
# DOMTree = xml.dom.minidom.parse(PROJECT_PATH + "/XML_resolver/data/SemEval2016_task3_test_input/train-more-for-subtaskA-from-2015/" + file_name)
DOMTree = xml.dom.minidom.parse(PROJECT_PATH + "/XML_resolver/data/SemEval2016_task3_test_input/test/" + file_name)
collection = DOMTree.documentElement

# movies = collection.getElementsByTagName("OrgQuestion")
movies = collection.getElementsByTagName("Thread")

count = 0
qcate_set =set()
for movie in movies:
    # Thread = movie.getElementsByTagName('Thread')
    Thread1 = movie.getElementsByTagName('RelQuestion')
    for iii in Thread1:
        # RelQuestion = Thread.getElementsByTagName('RelQuestion')
        if iii.hasAttribute("RELQ_CATEGORY"):
            count += 1
            qcate_set.add(iii.getAttribute("RELQ_CATEGORY"))
            print(count, ".", iii.getAttribute("RELQ_CATEGORY"))

print(file_name)
print("--------------")
print(qcate_set)
print("count:", len(qcate_set))


