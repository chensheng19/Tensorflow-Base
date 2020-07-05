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
#=====================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize":[],"loss":[]}
def moving_zverge(a,w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[idx-w:idx])//w for idx,val in enumerate(a)]

train_X = np.linspace(-1,1,100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()

tf.reset_default_graph()

#定义IP接口
strps_hosts = "localhost:1681"
strworker_hosts = "localhost:1682,localhost:1683"

#定义角色名称
strjob_name = "worker"
task_index = 1

#将字符串转化为数组
ps_hosts = strps_hosts.split(",")
worker_hosts = strworker_hosts.split(",")

#创建一个集群，对所有任务进行描述，对于所有任务的描述内容应该相同
cluster_spec = tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})

#为每个任务创建Server实例
server = tf.train.Server({'ps':ps_hosts,'worker':worker_hosts},
        job_name = strjob_name,task_index=task_index)

#ps角色采用server.join函数进行线程挂起，开始接受连接消息
if strjob_name == 'ps':
    print("wait")
    server.join()

#创建网络结构，将全部结点放在当前任务下
#通过tf.device中的任务通过tf.train.replica_device_setter函数来实现，第一个参数定义具体任务名称，第二个参数配置指定角色对应的IP地址
with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % task_index,cluster = cluster_spec)):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(tf.random_normal([1]),name="weight")
    b = tf.Variable(tf.zeros([1]),name="bias")

    global_step = tf.train.get_or_create_global_step() #创建或获取迭代次数张量

    z = tf.multiply(X,W)+b
    tf.summary.histogram('z',z) #将预测值以直方图形式显示

    cost = tf.reduce_mean(tf.square(Y-z))
    tf.summary.scalar('loss_function',cost) #将loss以标量显示
    learning_rate = 0.01
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1)
    merged_summary_op = tf.summary.merge_all() #定义合并所有summary的操作结点

    init = tf.global_variables_initializer()

training_epochs = 2200
display_step = 2

#创建Supervisor 管理session
sv = tf.train.Supervisor(is_chief=(task_index==0), #表明是否为chief supervisor
        logdir = "log/super/", #检查点文件和summary保存路径
        init_op = init, #初始化变量
        summary_op = None, #不自动保存summary
        saver = saver, #自动保存检查点文件
        global_step = global_step,
        save_model_secs = 5) # 保存检查点文件的时间间隔

#连接目标角色创建session
with sv.managed_session(server.target) as sess:
    print("sess ok")
    print(global_step.eval(session=sess))

    for epoch in range(global_step.eval(session=sess),training_epochs*len(train_X)):
        for (x,y) in zip(train_X,train_Y):
            _,epoch = sess.run([opt,global_step],feed_dict={X:x,Y:y})
            summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y}) #生成summary
            #sv.summary_computed(sess,summary_str,global_step=epoch) #将summary写入文件
    
            if epoch % display_step == 0:
                loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
                print("Epoch:",epoch+1,"cost:",loss,"W:",sess.run(W),"b:",sess.run(b))
                if not (loss=="NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)

    print("Finished!")
    sv.saver.save(sess,"log/mnist_with_summaries/"+"sv.cpk",global_step=epoch)

sv.stop
