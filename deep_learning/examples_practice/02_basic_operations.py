#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 11/5/18 4:53 PM
# @Author: Yongbo Wang
# @Email : yonwang@redhat.com
# @File  : 02_basic_operations.py
# @Desc  :

import tensorflow as tf

# ----------------
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a=2, b=3")
    print("add: %i" % sess.run(a + b))
    print("multi: %i" % sess.run(a * b))

# ----------------
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("2 add: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("2 multi: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# -----------------
# type --> tensor
matrix1 = tf.constant([[1, 2]])
matrix2 = tf.constant([[3], [4]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    # type --> numpy.ndarray
    result = sess.run(product)
    print("mul matrix: ", result)
    print(type(matrix1), type(result))