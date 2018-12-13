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

a = {'Working in Qatar', 'Family Life in Qatar', 'Environment', 'Sports in Qatar', 'Advice and Help', 'Moving to Qatar', 'Computers and Internet', 'Qatari Culture', 'Doha Shopping', 'Sightseeing and Tourist attractions', 'Language', 'Health and Fitness', 'Opportunities', 'Investment and Finance', 'Politics', 'Visas and Permits', 'Beauty and Style', 'Life in Qatar', 'Qatar Living Lounge', 'Electronics', 'Socialising', 'Salary and Allowances', 'Funnies', 'Pets and Animals', 'Education', 'Cars'}
b = {'Pets and Animals', 'Health and Fitness', 'Funnies', 'Education', 'Opportunities', 'Politics', 'Qatar Living Lounge', 'Cars', 'Qatari Culture', 'Working in Qatar', 'Beauty and Style', 'Environment', 'Sports in Qatar', 'Family Life in Qatar', 'Language', 'Electronics', 'Sightseeing and Tourist attractions', 'Moving to Qatar', 'Socialising', 'Life in Qatar', 'Doha Shopping', 'Investment and Finance', 'Computers and Internet', 'Advice and Help', 'Visas and Permits', 'Salary and Allowances'}
c = {'Cars', 'Visas and Permits', 'Family Life in Qatar', 'Opportunities', 'Qatari Culture', 'Investment and Finance', 'Sports in Qatar', 'Advice and Help', 'Qatar Living Lounge', 'Language', 'Sightseeing and Tourist attractions', 'Socialising', 'Funnies', 'Environment', 'Computers and Internet', 'Education', 'Health and Fitness', 'Life in Qatar', 'Salary and Allowances', 'Doha Shopping', 'Electronics', 'Pets and Animals', 'Working in Qatar', 'Beauty and Style', 'Moving to Qatar', 'Politics', 'Qatar 2022'}
d = {'Health and Fitness', 'Visas and Permits', 'Family Life in Qatar', 'Electronics', 'Moving to Qatar', 'Missing home!', 'Socialising', 'Sightseeing and Tourist attractions', 'Cars', 'Education', 'Qatar Living Lounge', 'Doha Shopping', 'Advice and Help', 'Language', 'Pets and Animals', 'Life in Qatar', 'Environment', 'Computers and Internet', 'Working in Qatar', 'Salary and Allowances', 'Investment and Finance'}

e = set()

for i in a:
    e.add(i)
for i in b:
    e.add(i)
for i in c:
    e.add(i)
for i in d:
    e.add(i)

print(e)
print(len(e))

