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

a = {'Beauty and Style', 'Welcome to Qatar', 'Computers and Internet', 'Advice and Help', 'Doha Shopping', 'Socialising', 'Pets and Animals', 'Opportunities', 'Investment and Finance', 'Language', 'Qatar Living Lounge', 'Working in Qatar', 'Cars and driving', 'Salary and Allowances', 'Qatari Culture', 'Politics', 'Missing home!', 'Environment', 'Family Life in Qatar', 'Visas and Permits', 'Health and Fitness', 'Qatar Living Tigers....', 'Moving to Qatar', 'Education', 'Sports in Qatar', 'Sightseeing and Tourist attractions', 'Electronics', 'Funnies'}
b = {'Moving to Qatar', 'Health and Fitness', 'Advice and Help', 'Politics', 'Cars and driving', 'Welcome to Qatar', 'Education', 'Opportunities', 'Salary and Allowances', 'Qatar Living Lounge', 'Visas and Permits', 'Sports in Qatar', 'Doha Shopping', 'Socialising', 'Investment and Finance', 'Pets and Animals', 'Funnies', 'Family Life in Qatar', 'Environment', 'Qatari Culture', 'Working in Qatar', 'Sightseeing and Tourist attractions', 'Missing home!'}
c = {'Advice and Help', 'Environment', 'Qatar 2022', 'Socialising', 'Pets and Animals', 'Beauty and Style', 'Cars', 'Politics', 'Qatar Living Lounge', 'Doha Shopping', 'Investment and Finance', 'Health and Fitness', 'Working in Qatar', 'Family Life in Qatar', 'Sports in Qatar', 'Life in Qatar', 'Sightseeing and Tourist attractions', 'Visas and Permits', 'Salary and Allowances', 'Language', 'Education', 'Computers and Internet', 'Qatari Culture', 'Opportunities', 'Electronics', 'Funnies', 'Moving to Qatar'}

m = set()

for i in a:
    m.add(i)
for i in b:
    m.add(i)
for i in c:
    m.add(i)

print(m)
print(len(m))

