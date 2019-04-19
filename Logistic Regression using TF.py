# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:51:45 2019

@author: GIASN
"""
import tensorflow as tf
from sklearn.datasets import make_moons
import numpy as np

learn_rate=0.01
minibatch=100
datasize=1000
noise_mag=0.01
n_epochs=500
n_inputs=2

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1)
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
y_moons_column_vector = y_moons.reshape(-1, 1)

test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


#Data_Train=make_moons(n_samples=datasize,noise=noise_mag,random_state=42)
#Data_Test=make_moons(n_samples=200,noise=noise_mag)
#reset_graph()

X=tf.placeholder(tf.float32, shape=(None, n_inputs+1), name= "X")
#Y=tf.placeholder(dtype=tf.int64,shape=None, name= "Y")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n_inputs+1, 1]), name="theta")
#b=tf.Variable(tf.zeros(shape=(1,1)),name="bias")
logits = tf.matmul(X, theta, name="logits")
#y_proba = 1 / (1 + tf.exp(-logits))

y_proba=tf.sigmoid(logits)


#W=tf.Variable(tf.random.uniform(shape=(Data_Train[0].shape[1],1)),name="weights")
#b=tf.Variable(tf.zeros(shape=(1,1)),name="bias")
"""
def logistic_regression(X,W,b):
    yhat=tf.sigmoid(tf.matmul(X,W)+b)
    return yhat
"""
#y_proba=tf.sigmoid(tf.matmul(X,W)+b)
"""
with tf.name_scope("loss"):
    #yhat=logistic_regression(X,W,b)
    mse=tf.losses.sigmoid_cross_entropy(multi_class_labels=Y,logits=yhat)
    loss=tf.reduce_mean(mse,name="loss")
"""

loss = tf.losses.log_loss(y, y_proba) 
   
with tf.name_scope("optimizer"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    training_op=optimizer.minimize(loss)
    
init=tf.global_variables_initializer()  


saver= tf.train.Saver()

n_batches = int(np.ceil(datasize / minibatch))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, minibatch)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)

    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
    
    savepath=saver.save(sess,"./Ch9_Assign1_final.ckpt")
    

    
    
