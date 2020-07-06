#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-07-05
# File name   : 
# Description : 
#
#======================================================
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import pylab

tf.reset_default_graph()

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

train_epochs = 20
batch_size = 100
display_step = 1

saver = tf.train.Saver()
model_path = "log/521model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(train_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([opt,cost],feed_dict={x:batch_x,y:batch_y})
            avg_cost += c/total_batch

        if (epoch+1)%display_step == 0:
            print("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("Finished!")

    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Acc:",acc.eval({x:mnist.test.images,y:mnist.test.labels}))

    save_path = saver.save(sess,model_path)
    print("Model saved in file:%s"%save_path)


print("Starting 2nd session!")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,model_path)

    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    print("Acc:",acc.eval({x:mnist.test.images,y:mnist.test.labels}))
