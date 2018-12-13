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

a = {'Environment', 'Socialising', 'Moving to Qatar', 'Pets and Animals', 'Sports in Qatar', 'Working in Qatar', 'Visas and Permits', 'Salary and Allowances', 'Missing home!', 'Welcome to Qatar', 'Beauty and Style', 'Family Life in Qatar', 'Cars and driving', 'Qatari Culture', 'Education', 'Opportunities', 'Doha Shopping', 'Qatar Living Lounge', 'Qatar Living Tigers....', 'Politics', 'Sightseeing and Tourist attractions', 'Computers and Internet', 'Language', 'Health and Fitness', 'Funnies', 'Advice and Help', 'Electronics'}
b = {'Working in Qatar', 'Advice and Help', 'Sports in Qatar', 'Computers and Internet', 'Cars and driving', 'Family Life in Qatar', 'Beauty and Style', 'Visas and Permits', 'Salary and Allowances', 'Politics', 'Language', 'Socialising', 'Funnies', 'Welcome to Qatar', 'Sightseeing and Tourist attractions', 'Moving to Qatar', 'Opportunities', 'Investment and Finance', 'Education', 'Health and Fitness', 'Qatari Culture', 'Doha Shopping', 'Qatar Living Lounge', 'Environment'}
c = {'Opportunities', 'Sightseeing and Tourist attractions', 'Politics', 'Visas and Permits', 'Education', 'Qatar Living Tigers....', 'Salary and Allowances', 'Doha Shopping', 'Welcome to Qatar', 'Qatari Culture', 'Health and Fitness', 'Family Life in Qatar', 'Funnies', 'Working in Qatar', 'Missing home!', 'Qatar Living Lounge', 'Environment', 'Moving to Qatar', 'Advice and Help', 'Pets and Animals', 'Language', 'Beauty and Style', 'Cars and driving', 'Computers and Internet', 'Socialising', 'Electronics', 'Sports in Qatar'}
d = {'Cars and driving', 'Funnies', 'Welcome to Qatar', 'Doha Shopping', 'Sports in Qatar', 'Moving to Qatar', 'Environment', 'Sightseeing and Tourist attractions', 'Family Life in Qatar', 'Qatari Culture', 'Beauty and Style', 'Advice and Help', 'Computers and Internet', 'Health and Fitness', 'Politics', 'Language', 'Investment and Finance', 'Socialising', 'Salary and Allowances', 'Opportunities', 'Working in Qatar', 'Qatar Living Lounge', 'Visas and Permits', 'Education'}
e = {'Funnies', 'Qatar Living Lounge', 'Welcome to Qatar', 'Advice and Help', 'Health and Fitness', 'Family Life in Qatar', 'Computers and Internet', 'Language', 'Missing home!', 'Beauty and Style', 'Electronics', 'Education', 'Politics', 'Qatar Living Tigers....', 'Salary and Allowances', 'Opportunities', 'Working in Qatar', 'Socialising', 'Visas and Permits', 'Cars and driving', 'Pets and Animals', 'Qatari Culture', 'Sports in Qatar', 'Sightseeing and Tourist attractions', 'Moving to Qatar', 'Environment', 'Doha Shopping'}
f = {'Politics', 'Qatar Living Lounge', 'Sightseeing and Tourist attractions', 'Socialising', 'Salary and Allowances', 'Family Life in Qatar', 'Funnies', 'Beauty and Style', 'Health and Fitness', 'Opportunities', 'Language', 'Investment and Finance', 'Working in Qatar', 'Moving to Qatar', 'Advice and Help', 'Doha Shopping', 'Welcome to Qatar', 'Environment', 'Visas and Permits', 'Sports in Qatar', 'Education', 'Qatari Culture', 'Computers and Internet', 'Cars and driving'}
g = {'Working in Qatar', 'Visas and Permits', 'Sightseeing and Tourist attractions', 'Welcome to Qatar', 'Sports in Qatar', 'Funnies', 'Opportunities', 'Salary and Allowances', 'Socialising', 'Doha Shopping', 'Missing home!', 'Family Life in Qatar', 'Qatari Culture', 'Cars and driving', 'Moving to Qatar', 'Qatar Living Tigers....', 'Health and Fitness', 'Qatar Living Lounge', 'Electronics', 'Beauty and Style', 'Advice and Help', 'Education', 'Computers and Internet', 'Environment', 'Language', 'Pets and Animals', 'Politics'}
h = {'Beauty and Style', 'Visas and Permits', 'Moving to Qatar', 'Cars and driving', 'Computers and Internet', 'Language', 'Advice and Help', 'Salary and Allowances', 'Welcome to Qatar', 'Qatari Culture', 'Opportunities', 'Sightseeing and Tourist attractions', 'Environment', 'Doha Shopping', 'Family Life in Qatar', 'Funnies', 'Sports in Qatar', 'Politics', 'Health and Fitness', 'Education', 'Working in Qatar', 'Socialising', 'Qatar Living Lounge', 'Investment and Finance'}

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

print(m)
print(len(m))

