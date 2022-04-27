# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#  tf.disable_v2_behavior()

# create data
x_data = np.random.rand(100).astype(np.float32)     # 一百个随机数列  定义数据类型
y_data = x_data*0.1+0.3          # W为0.3
# print(y_data)
# create tensorflow structure start #
# AttributeError: module 'tensorflow' has no attribute 'random_uniform'
# V2版本里面random_uniform改为random.uniform
Weights = tf.Variable(tf.random.uniform((1,), -1.0, 1.0))  # 随机数列生产的参数 [1] W结构为一维 -1.0, 1.0 随机生产数列的范围
# TypeError: 'function' object is not subscriptable
# 一般是少了括号
print(Weights)
biases = tf.Variable(tf.zeros((1,)))     # 定义初始值 0
print(biases)
# y = Weights*x_data+biases
# 定义预测值y

def loss():
    w = Weights * x_data + biases
    print(tf.keras.losses.MSE(y_data, w))

    return tf.keras.losses.MSE(y_data, w)  # alias 计算loss  预测值与真实值的差别
# AttributeError: module 'tensorflow_core._api.v2.train' has no attribute 'GradientDescentOptimizer'
# "tf.train.GradientDescentOptimizer" change "tf.compat.v1.train.GradientDescentOptimizer"
# `loss` passed to Optimizer.compute_gradients should be a function when eager execution is enabled.
# 神经网络知道误差以后 优化器（optimizer）减少误差 提升参数的准确度
# 其中的minimize可以拆为以下两个步骤：
# ① 梯度计算
# ② 将计算出来的梯度应用到变量的更新中
# 拆开的好处是，可以对计算的梯度进行限制，防止梯度消失和爆炸
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)  # alias: tf.optimizers.SGD learning_rate=0.5
#init = tf.initializer_all_variables()
#  create tensorflow structure end #

# create session
# sess = tf.Session()
# sess.run(init)          # Very important

for step in range(201):
    optimizer.minimize(loss,var_list=[Weights,biases])
    if step % 20 == 0:
        print("{} step, Weights = {}, biases = {}"
              .format(step, Weights.read_value(), biases.read_value()))  # read_value函数可用numpy替换

