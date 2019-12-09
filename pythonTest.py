import tensorflow as tf
import numpy as np
import random 
import cv2
import pandas as pd
from sklearn.preprocessing import Normalizer
pf = pd.DataFrame([[1,2],[3,4]],columns=["A","B"])
print(pf)
# xx = {"a":1}
# v1 = tf.constant([1.,1.,0.,0.])
# v2 = tf.constant([1.,0.,1.,0.])
# v3 = tf.reduce_sum(tf.add(v1,v2))
# with tf.Session() as sess:
#     print(sess.run(v3))
# print(cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_L1, dtype=cv2.CV_32F))
# v1 = tf.cast(tf.constant([1.,1.,0.,0.]),dtype=tf.bool)
# v2 = tf.cast(tf.constant([1.,0.,1.,0.]),dtype=tf.bool)
# v3 = tf.logical_and(v1,v2)
# v4 = tf.logical_or(v1,v2)
# v5 = tf.reduce_sum(tf.cast(v3,tf.float32))
# v6 = tf.reduce_sum(tf.cast(v4,tf.float32))
# tx = np.array([[1,2],[4,3]])
# outputs = []
# for i in range(2):
#     outputs.append(["T1","T2"])
# outputs = list(zip(*outputs))
# print(outputs)
# names = ["A","N"]
# for i,o in zip(outputs,names):
#     print(i)
#     print(o)
# for i in range(2):
#     print(i)
# with tf.Session() as sess:
#     print(sess.run(v5 / v6))
