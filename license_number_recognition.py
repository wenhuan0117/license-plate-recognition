import input_data
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
##
xs=tf.placeholder(tf.float32,[None,784],name='inputs')/255.
ys=tf.placeholder(tf.float32,[None,10],name='outputs')

def compute_accuracy(x, y):
    with tf.Session() as sess:
        correct_prediction = tf.equal(x, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy)
    return result

def normalization(inputs):
    mean,var=tf.nn.moments(inputs,axes=[0])
    shape1=inputs.get_shape()[1].value
    scale=tf.Variable(tf.ones([shape1]))
    shift=tf.Variable(tf.zeros([shape1]))
    epsilon=0.001
##    out1=(inputs-mean)/tf.sqrt(var+epsilon)
##    out=scale*out1+shift
    out=tf.nn.batch_normalization(inputs,mean,var,shift,scale,epsilon)
    return out

def add_layers(xs1,out_size,name=None,activation=None,normalization=None):
    input_size=xs1.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        weights_inia=tf.truncated_normal([input_size[-1],out_size],stddev=0.1)
        biases_inia=tf.constant(0.1,shape=[out_size])
        weights=tf.get_variable('weights',initializer=weights_inia)
        biases=tf.get_variable('biases',initializer=biases_inia)

    out1=tf.matmul(xs1,weights)+biases
    if activation is None:
        pass
    else:
        out1=activation(out1)

    if normalization is not None:
        out1=normalization(out1)
        
    return out1

def conv2d(x,insize,outsize,name=None):
    with tf.variable_scope(name) as scope:
        W_ini=tf.truncated_normal([5,5,insize,outsize],stddev=0.1)
        b_ini=tf.constant(0.1,shape=[outsize])
        W=tf.get_variable('w',initializer=W_ini)
        b=tf.get_variable('b',initializer=b_ini)
        conv1=tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
        result=tf.nn.relu(conv1+b)
    return result

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x_image = tf.reshape(xs, [-1, 28, 28, 1])

h_conv1 = conv2d(x_image,1,32,name='conv1')

h_pool1=max_pool_2x2(h_conv1)

h_conv2=conv2d(h_pool1,32,64,name='conv2')

h_pool2=max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

output1=add_layers(h_pool2_flat,1024,name='fc1',activation=tf.nn.relu,normalization=None)

output2=add_layers(output1,10,name='fc2',activation=None,normalization=None)

prediction=tf.nn.softmax(output2)

pre=tf.argmax(prediction,1)

##loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
cross_entry=tf.nn.softmax_cross_entropy_with_logits(logits=output2, labels=ys)
loss=tf.reduce_mean(cross_entry)

save_net_path='./my_mnist/net'


def train_net():
    LR=0.01
    LR_DECY=0.99
    global_step=tf.Variable(0,trainable=False)
    learn_rate=tf.train.exponential_decay(LR,global_step,20,LR_DECY)
    train_op=tf.train.GradientDescentOptimizer(learn_rate).minimize(loss,global_step=global_step)
##    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(1000):
            batch=mnist.train.next_batch(100)
            sess.run(train_op,feed_dict={xs:batch[0],ys:batch[1]})

            if i%50==0:
                pre1,los,_=sess.run([pre,loss,train_op],feed_dict={xs:batch[0],ys:batch[1]})
                ori=sess.run(tf.argmax(batch[1],1))
                print(compute_accuracy(pre1, ori))
        saver.save(sess,save_net_path)

def pre_net(x_data):
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,save_net_path)
        pre_re,out_re=sess.run([pre,prediction],feed_dict={xs:x_data})
        return pre_re,out_re

##
##train_net()


import license_plate_find
import cv2

def cp_number_detect(img):
    cp=license_plate_find.find_cp(img)
    cp_number=license_plate_find.detect_cpnum(cp)
        
    def image_data(img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        shape=[int(binary.shape[0]*1.8),int(binary.shape[1]*1.8)]
        image_big=np.full(shape,0,dtype=np.float32)
        image_big[int(binary.shape[0]*0.4):int(binary.shape[0]*1.4),int(binary.shape[1]*0.4):int(binary.shape[1]*1.4)]=binary

        image=cv2.resize(image_big,(28,28))
        cv2.imshow('image_num',image_big)
        image1=np.reshape(image,(1,784))
        return image1

    for i,num_tobe_test in enumerate(cp_number):
        x_input=image_data(num_tobe_test)/255.
        pre_re,out_re=pre_net(x_input)
        print(pre_re)
        
    cv2.imshow('image',img)

img=cv2.imread('cp.jpg')    
cp_number_detect(img)

'''
batch=mnist.train.next_batch(1)
image1=np.reshape(batch[0],(28,28))
image2=image1
cv2.imshow('batch',image1)

pre_re,out_re=pre_net(batch[0])
print(pre_re)
'''
