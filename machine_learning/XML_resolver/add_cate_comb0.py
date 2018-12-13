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

a = {'Environment', 'Education', 'Language', 'Socialising', 'Sports in Qatar', 'Salary and Allowances', 'Investment and Finance', 'Advice and Help', 'Qatar Living Lounge', 'Pets and Animals', 'Computers and Internet', 'Qatari Culture', 'Doha Shopping', 'Electronics', 'Visas and Permits', 'Qatar 2022', 'Missing home!', 'Opportunities', 'Family Life in Qatar', 'Sightseeing and Tourist attractions', 'Working in Qatar', 'Health and Fitness', 'Funnies', 'Politics', 'Beauty and Style', 'Moving to Qatar', 'Life in Qatar', 'Cars'}
# 1.文件夹“train”中：
b = {'Beauty and Style', 'Welcome to Qatar', 'Computers and Internet', 'Advice and Help', 'Doha Shopping', 'Socialising', 'Pets and Animals', 'Opportunities', 'Investment and Finance', 'Language', 'Qatar Living Lounge', 'Working in Qatar', 'Cars and driving', 'Salary and Allowances', 'Qatari Culture', 'Politics', 'Missing home!', 'Environment', 'Family Life in Qatar', 'Visas and Permits', 'Health and Fitness', 'Qatar Living Tigers....', 'Moving to Qatar', 'Education', 'Sports in Qatar', 'Sightseeing and Tourist attractions', 'Electronics', 'Funnies'}
#
# b = {'Family Life in Qatar', 'Qatar Living Lounge', 'Qatari Culture', 'Language', 'Funnies', 'Cars and driving', 'Socialising', 'Advice and Help', 'Qatar 2022', 'Investment and Finance', 'Sports in Qatar', 'Environment', 'Visas and Permits', 'Life in Qatar', 'Welcome to Qatar', 'Salary and Allowances', 'Politics', 'Education', 'Electronics', 'Doha Shopping', 'Opportunities', 'Computers and Internet', 'Moving to Qatar', 'Cars', 'Health and Fitness', 'Beauty and Style', 'Missing home!', 'Sightseeing and Tourist attractions', 'Qatar Living Tigers....', 'Pets and Animals', 'Working in Qatar'}

m = set()

for i in a:
    m.add(i)
for i in b:
    m.add(i)

print(m)
print(len(m))

