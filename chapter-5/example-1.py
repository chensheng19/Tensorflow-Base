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
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

tf.reset_default_graph() #清空默认图

x = tf.placeholder(tf.float32,[None,784]) #定义占位符
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10])) #定义参数变量
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,W)+b) #定义前向计算

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1)) #定义损失函数

learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #定义优化操作，优化损失函数

train_epochs = 20
batch_size = 100
display_step = 1

saver = tf.train.Saver() #创建一个saver实例
model_path = "log/521model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #初始化参数

    for epoch in range(train_epochs): #迭代每个epoch
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size) #计算batch数量

        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size) #依次获取每个batch数据
            _,c = sess.run([opt,cost],feed_dict={x:batch_x,y:batch_y}) #运行优化操作结点和损失函数结点
            avg_cost += c/total_batch #计算平均损失函数

        if (epoch+1)%display_step == 0:
            print("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("Finished!")

    #作预测
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1)) #将预测正确标记为True，错误标记为False
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #将布尔值转化为0和1，计算平均值，即为准确率
    print("Acc:",acc.eval({x:mnist.test.images,y:mnist.test.labels}))
    
    #模型保存
    save_path = saver.save(sess,model_path)
    print("Model saved in file:%s"%save_path)

    #模型加载
print("Starting 2nd session!")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,model_path)

    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    print("Acc:",acc.eval({x:mnist.test.images,y:mnist.test.labels}))
