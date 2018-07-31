# encoding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""read the data"""
data_mni = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""x : data for train , y_ : labels of data"""
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

"""w :权重 ,b:偏置, y:预测类别: y = softmax(wx+b)"""
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w)+b)

"""cross_entropy: loss 损失,目标类别和预测类别之间的交叉熵,train_step:使用梯度下降来更新参数"""
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

"""评估模型:correct_prediction bool tensor  tf.cast : bool to float  accuracy:accuracy rate"""
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

""" run the 运行的计算图"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    """迭代2000train_step to 训练模型"""
    for i in range(2000):
        batch = data_mni.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    """accuracy rate of test"""
    print "accuracy rate of test: %.2f%%" % sess.run(accuracy * 100, feed_dict={x: data_mni.test.images, y_: data_mni.test.labels})
    """predict top 5 of test_data labels"""
    print "predict labels: ", sess.run(tf.argmax(y, 1), feed_dict={x: data_mni.test.images[:5]})
    print "the true labels:", tf.argmax(data_mni.test.labels[:5], 1).eval()
