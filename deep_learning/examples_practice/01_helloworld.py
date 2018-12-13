#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 11/5/18 2:31 PM
# @Author: Yongbo Wang
# @Email : yonwang@redhat.com
# @File  : 01_helloworld.py
# @Desc  :

import tensorflow as tf


hello = tf.constant('Hello World!')
sess = tf.Session()

print(sess.run(hello))