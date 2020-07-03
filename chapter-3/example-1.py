#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-07-02
# File name   : 
# Description : 
#
#=====================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1,1,100) #创建-1到1的等差数列，含100个元素
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 #生成y数据，并添加噪音，*表示传入的是一个元组

plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()

X = tf.placeholder("float") #创建输入输出占位符
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]),name="weight") #定义参数变量
b = tf.Variable(tf.zeros([1]),name="bias")

z = tf.multiply(X,W)+b #定义运算

cost = tf.reduce_mean(tf.square(Y-z)) #定义损失函数
learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #定义优化操作

init = tf.global_variables_initializer()
display_step=2
train_epochs=20

def moving_averge(a,w=10):
    if len(a) < w:
        return a[:] #当epoch小于10次时，直接返回loss
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)] #当epoch大于10时，前10个loss直接返回，后面的loss累加并除以10

with tf.Session() as sess:
    sess.run(init) #初始化所有变量

    plotdata = {"batchsize":[],"loss":[]}

    for epoch in range(train_epochs):
        for x,y in zip(train_X,train_Y):
            sess.run(opt,feed_dict={X:x,Y:y}) #运算优化操作结点，优化参数

            if epoch % display_step == 0:
                loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y}) #计算当前epoch的损失函数值
                print("Epoch:",epoch+1,"loss=",loss,"W=",sess.run(W),"b=",sess.run(b))

                if not loss == 'NA':
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)

    print("Finished!")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))

    plt.plot(train_X,train_Y,'ro',label='Original') #绘制原始数据
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted') #绘制预测数据
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_averge(plotdata["loss"]) #计算移动平均损失
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--') #计算每个epoch的移动平均损失
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run VS. Training loss')
    plt.show()
