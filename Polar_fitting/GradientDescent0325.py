'''来源 https://blog.csdn.net/weixin_39794340/article/details/110700424'''
import numpy as np
def featureNormalize(X):
    X_norm = X;
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    for i in range(X.shape[1]):
        mu[0,i] = np.mean(X[:,i]) # 均值
        sigma[0,i] = np.std(X[:,i]) # 标准差
        # print(mu)
        # print(sigma)
    X_norm = (X - mu) / sigma
    return X_norm,mu,sigma
#计算损失
def computeCost(X, y, theta):
    m = y.shape[0]
    # J = (np.sum((X.dot(theta) - y)**2)) / (2*m)
    C = X.dot(theta) - y
    J2 = (C.T.dot(C))/ (2*m)
    return J2

#梯度下降:
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    #print(m)
    # 存储历史误差
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        # 对J求导，得到 alpha/m * (WX - Y)*x(i)， (3,m)*(m,1) X (m,3)*(3,1) = (m,1)
        theta = theta - (alpha/m) * (X.T.dot(X.dot(theta) - y))
        J_history[iter] = computeCost(X, y, theta)
    return J_history,theta
