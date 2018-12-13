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

a = {'Computers and Internet', 'Politics', 'Funnies', 'Language', 'Qatari Culture', 'Education', 'Moving to Qatar', 'Cars', 'Qatar Living Lounge', 'Pets and Animals', 'Beauty and Style', 'Sightseeing and Tourist attractions', 'Sports in Qatar', 'Environment', 'Health and Fitness', 'Electronics', 'Socialising', 'Investment and Finance', 'Doha Shopping', 'Working in Qatar', 'Salary and Allowances', 'Life in Qatar', 'Family Life in Qatar', 'Opportunities', 'Visas and Permits', 'Advice and Help'}
b = {'Salary and Allowances', 'Electronics', 'Socialising', 'Funnies', 'Investment and Finance', 'Computers and Internet', 'Beauty and Style', 'Cars', 'Education', 'Environment', 'Moving to Qatar', 'Doha Shopping', 'Sports in Qatar', 'Visas and Permits', 'Qatar Living Lounge', 'Politics', 'Health and Fitness', 'Qatari Culture', 'Sightseeing and Tourist attractions', 'Life in Qatar', 'Advice and Help', 'Working in Qatar', 'Opportunities', 'Pets and Animals', 'Family Life in Qatar', 'Language'}
c = {'Salary and Allowances', 'Language', 'Health and Fitness', 'Cars', 'Qatari Culture', 'Sightseeing and Tourist attractions', 'Investment and Finance', 'Family Life in Qatar', 'Socialising', 'Life in Qatar', 'Environment', 'Working in Qatar', 'Moving to Qatar', 'Visas and Permits', 'Funnies', 'Politics', 'Education', 'Beauty and Style', 'Qatar Living Lounge', 'Sports in Qatar', 'Opportunities', 'Advice and Help', 'Electronics', 'Pets and Animals', 'Doha Shopping', 'Computers and Internet'}
d = {'Computers and Internet', 'Beauty and Style', 'Electronics', 'Visas and Permits', 'Health and Fitness', 'Advice and Help', 'Cars', 'Qatari Culture', 'Education', 'Sightseeing and Tourist attractions', 'Pets and Animals', 'Politics', 'Life in Qatar', 'Qatar 2022', 'Family Life in Qatar', 'Socialising', 'Investment and Finance', 'Salary and Allowances', 'Sports in Qatar', 'Moving to Qatar', 'Opportunities', 'Qatar Living Lounge', 'Working in Qatar', 'Environment', 'Doha Shopping', 'Language', 'Funnies'}
e = {'Advice and Help', 'Beauty and Style', 'Visas and Permits', 'Electronics', 'Health and Fitness', 'Investment and Finance', 'Politics', 'Sports in Qatar', 'Computers and Internet', 'Family Life in Qatar', 'Environment', 'Salary and Allowances', 'Pets and Animals', 'Life in Qatar', 'Education', 'Moving to Qatar', 'Working in Qatar', 'Funnies', 'Socialising', 'Sightseeing and Tourist attractions', 'Qatari Culture', 'Cars', 'Qatar Living Lounge', 'Opportunities', 'Doha Shopping'}
f = {'Visas and Permits', 'Health and Fitness', 'Salary and Allowances', 'Electronics', 'Sports in Qatar', 'Beauty and Style', 'Funnies', 'Working in Qatar', 'Socialising', 'Moving to Qatar', 'Computers and Internet', 'Opportunities', 'Pets and Animals', 'Life in Qatar', 'Environment', 'Politics', 'Family Life in Qatar', 'Advice and Help', 'Education', 'Investment and Finance', 'Qatari Culture', 'Sightseeing and Tourist attractions', 'Cars', 'Doha Shopping', 'Qatar Living Lounge'}
g = {'Sports in Qatar', 'Health and Fitness', 'Funnies', 'Opportunities', 'Computers and Internet', 'Electronics', 'Life in Qatar', 'Investment and Finance', 'Politics', 'Cars', 'Visas and Permits', 'Doha Shopping', 'Qatari Culture', 'Beauty and Style', 'Education', 'Family Life in Qatar', 'Environment', 'Sightseeing and Tourist attractions', 'Socialising', 'Moving to Qatar', 'Pets and Animals', 'Qatar Living Lounge', 'Advice and Help', 'Salary and Allowances', 'Working in Qatar'}
h = {'Family Life in Qatar', 'Advice and Help', 'Pets and Animals', 'Beauty and Style', 'Environment', 'Qatari Culture', 'Electronics', 'Working in Qatar', 'Sports in Qatar', 'Opportunities', 'Computers and Internet', 'Language', 'Cars', 'Visas and Permits', 'Sightseeing and Tourist attractions', 'Moving to Qatar', 'Funnies', 'Salary and Allowances', 'Doha Shopping', 'Socialising', 'Politics', 'Health and Fitness', 'Qatar Living Lounge', 'Education', 'Life in Qatar', 'Qatar 2022', 'Investment and Finance'}
n = {'Electronics', 'Beauty and Style', 'Investment and Finance', 'Doha Shopping', 'Language', 'Cars', 'Environment', 'Qatari Culture', 'Health and Fitness', 'Moving to Qatar', 'Working in Qatar', 'Family Life in Qatar', 'Life in Qatar', 'Education', 'Sports in Qatar', 'Advice and Help', 'Pets and Animals', 'Politics', 'Qatar 2022', 'Opportunities', 'Funnies', 'Sightseeing and Tourist attractions', 'Qatar Living Lounge', 'Socialising', 'Computers and Internet', 'Visas and Permits', 'Salary and Allowances'}

m = set()

for i in a:
    m.add(i)
for i in b:
    m.add(i)
for i in c:
    m.add(i)
for i in d:
    m.add(i)
for i in e:
    m.add(i)
for i in f:
    m.add(i)
for i in g:
    m.add(i)
for i in h:
    m.add(i)
for i in n:
    m.add(i)

print(m)
print(len(m))

