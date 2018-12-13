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

a = {'Working in Qatar', 'Missing home!', 'Advice and Help', 'Moving to Qatar', 'Health and Fitness', 'Socialising', 'Family Life in Qatar', 'Pets and Animals', 'Welcome to Qatar', 'Sports in Qatar', 'Education', 'Salary and Allowances', 'Qatar Living Lounge', 'Doha Shopping', 'Investment and Finance', 'Qatari Culture', 'Visas and Permits', 'Politics', 'Funnies', 'Sightseeing and Tourist attractions', 'Opportunities', 'Cars and driving', 'Environment'}
b = {'Working in Qatar', 'Doha Shopping', 'Moving to Qatar', 'Family Life in Qatar', 'Qatar Living Lounge', 'Pets and Animals', 'Socialising', 'Missing home!', 'Salary and Allowances', 'Politics', 'Visas and Permits', 'Advice and Help', 'Sports in Qatar', 'Cars and driving', 'Opportunities', 'Welcome to Qatar', 'Environment', 'Sightseeing and Tourist attractions', 'Education', 'Funnies', 'Qatari Culture'}
c = {'Cars and driving', 'Welcome to Qatar', 'Funnies', 'Pets and Animals', 'Moving to Qatar', 'Qatar Living Lounge', 'Working in Qatar', 'Salary and Allowances', 'Missing home!', 'Politics', 'Sports in Qatar', 'Opportunities', 'Family Life in Qatar', 'Qatari Culture', 'Sightseeing and Tourist attractions', 'Socialising', 'Education', 'Doha Shopping', 'Environment', 'Advice and Help', 'Visas and Permits'}
d = {'Salary and Allowances', 'Working in Qatar', 'Socialising', 'Politics', 'Missing home!', 'Moving to Qatar', 'Qatar Living Lounge', 'Sightseeing and Tourist attractions', 'Advice and Help', 'Environment', 'Opportunities', 'Education', 'Visas and Permits', 'Sports in Qatar', 'Investment and Finance', 'Pets and Animals', 'Family Life in Qatar', 'Funnies', 'Doha Shopping', 'Qatari Culture', 'Health and Fitness', 'Welcome to Qatar', 'Cars and driving'}

m = set()

for i in a:
    m.add(i)
for i in b:
    m.add(i)
for i in c:
    m.add(i)
for i in d:
    m.add(i)

print(m)
print(len(m))

