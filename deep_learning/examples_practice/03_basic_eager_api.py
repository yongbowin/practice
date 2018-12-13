#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 11/5/18 5:20 PM
# @Author: Yongbo Wang
# @Email : yonwang@redhat.com
# @File  : 03_basic_eager_api.py
# @Desc  :

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

# set eager model
tfe.enable_eager_execution()

a = tf.constant(2)
b = tf.constant(3)

# operation without session run
c = a + b
d = a * b
print(a)
print(type(a))
print(c)
print(type(c))
print(d)
print(type(d))

a = tf.constant([[1, 2.], [2, 3]], dtype=tf.float32)
b = np.array([[2., 2], [3, 3]], dtype=np.float32)

print(a)
print(type(a))
print(b)
print(type(b))

c = a + b
print(c)
print(type(c))

d = tf.matmul(a, b)
print(d)
print(type(d))

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])