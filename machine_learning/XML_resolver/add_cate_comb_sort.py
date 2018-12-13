#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: Machine-Learning-Practice - add_cate.py
@time: 6/15/18 1:29 AM
@description: 
"""
import numpy as np

a = {'Opportunities', 'Pets and Animals', 'Salary and Allowances', 'Doha Shopping', 'Cars and driving', 'Environment', 'Funnies', 'Family Life in Qatar', 'Sports in Qatar', 'Qatar Musicians', 'Working in Qatar', 'Investment and Finance', 'Welcome to Qatar', 'Socialising', 'Visas and Permits', 'Moving to Qatar', 'Politics', 'Advice and Help', 'Sightseeing and Tourist attractions', 'Qatari Culture', 'Qatar Living Lounge '}
# count: 25

ee = np.sort(list(a))

print(ee)
