#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-07-03
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

strps_hosts = "localhost:1681"
strworker_hosts = "localhost:1682,localhost:1683"

strjob_name = "ps"
task_index = 0

ps_hosts = strps_hosts.split(",")
worker_hosts = strworker_hosts.split(",")
cluster_spec = tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})

server = tf.train.Server({'ps':ps_hosts,'worker':worker_hosts},
        job_name = strjob_name,task_index=task_index)

if strjob_name == 'ps':
    print("wait")
    server.join()

with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % task_index,cluster = cluster_spec)):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(tf.random_normal([1]),name="weight")
    b = tf.Variable(tf.zeros([1]),name="bias")

    global_step = tf.train.get_or_create_global_step()

    z = tf.multiply(X,W)+b
    tf.summary.histogram('z',z)

    cost = tf.reduce_mean(tf.square(Y-z))
    tf.summary.scalar('loss_function',cost)
    learning_rate = 0.01
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1)
    merged_summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

training_epochs = 2200
display_step = 2

sv = tf.train.Supervisor(is_chief=(task_index==0),
        logdir = "log/super/",
        init_op = init,
        summary_op = None,
        saver = saver,
        global_step = global_step,
        save_model_secs = 5)

with sv.managed_session(server.target) as sess:
    print("sess ok")
    print(global_step.eval(session=sess))

    for epoch in range(global_step.eval(session=sess),training_epochs*len(train_X)):
        _,epoch = sess.run([opt,global_step],feed_dict={X:x,Y:y})
        summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
        sv.summary_computed(sess,summary_str,global_step=epoch)

        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch:",epoch+1,"cost:",loss,"W:",sess.run(W),"b:",sess.run(b))
            if not (loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print("Finished!")
    sv.saver.save(sess,"log/mnist_with_summaries/"+"sv.cpk",global_step=epoch)

sv.stop

