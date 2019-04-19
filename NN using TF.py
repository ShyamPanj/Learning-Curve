# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

n_inputs=28*28

n_hidden1=300

n_hidden2=100

n_outputs=10

X=tf.placeholder(dtype=tf.float32,name="X",shape=(None,n_inputs))
Y=tf.placeholder(dtype=tf.int64,name="Y",shape=(None))

def neuron_layer(X, n_neurons,name, activation=None):
    with tf.name_scope(name):
        n_inputs=int(X.get_shape()[1])
        stddev=2/np.sqrt(n_inputs)
        init=tf.truncated_normal(shape=(n_inputs,n_neurons),stddev=stddev)
        W=tf.Variable(init,name="W")
        b=tf.Variable(tf.zeros(shape=n_neurons),name="biases")
        z=tf.matmul(X,W)+b
        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z
        
with tf.name_scope("dnn"):
    hidden1=neuron_layer(X,n_neurons=n_hidden1,activation="relu",name="hidden1")
    hidden2=neuron_layer(hidden1,n_hidden2,"hidden2",activation="relu")
    logits=neuron_layer(hidden2,n_outputs,"outputs")

from tensorflow.contrib.layers import fully_connected

with tf.name_scope("dnn"):
    hidden1=fully_connected(X,n_hidden1, scope="hidden1")
    hidden2=fully_connected(hidden1,n_hidden2,scope="hidden2")
    logits=fully_connected(hidden2,n_outputs,scope="outputs") 

with tf.name_scope("loss"):
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)
    loss= tf.reduce_mean(xentropy, name="loss")

learning_rate=0.1

with tf.name_scope("train"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)
  
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,Y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))


init=tf.global_variables_initializer()

saver=tf.train.Saver()


from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data")

n_epochs=400
batch_size=50
"""
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch,y_batch=mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X:X_batch,Y: y_batch})
            
        acc_train=accuracy.eval(feed_dict={X:X_batch,Y:y_batch})
        acc_test=accuracy.eval(feed_dict={X:mnist.test.images,Y: mnist.test.labels})
        print (epoch, "Training accuracy:", acc_train, "Test_accuracy:", acc_test)
    save_path= saver.save(sess,"./TF_1_final.ckpt")
##  
"""
with tf.Session() as sess:
    saver.restore(sess,"./TF_1_final.ckpt")   
    Z=logits.eval(feed_dict={X: mnist.test.images})
    y_pred=np.argmax(Z,axis=1)          
       
        
