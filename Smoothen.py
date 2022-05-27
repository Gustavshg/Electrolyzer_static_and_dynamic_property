import numpy as np
import pywt#python wavelet transform,还需要安装PyWavelets


"""这里是各种用来平滑的函数"""
def EMA(data,beta=0.5):#进行指数滑动平均，进行滤波
    theta = np.array(data).reshape(len(data),1)
    v = np.zeros((len(data),1))
    v[0] = theta[0]
    for t in np.arange(1,len(data)):
        v[t] = v[t-1]*beta + (1-beta)*theta[t]
    return v

def AA(data,step = 1):#算术平均滤波法
    v = list(data)
    for i in range(step,len(data) - step):
        v[i] = sum(data[i-step : i+step])/2/step
    # for t in np.arange(step+1):
    #     v[t] = np.average(data[:t+step])
    #     v[-t] = np.average(data[-(t+step):])
    # for t in np.arange(step+1,len(data)-step+1):
    #     v[t] = np.average(data[(t-step):(t+step)])

    return np.array(v)

def WL(data_input,threshold = 0.3):#小波分解
    index = []
    data = []
    for i in range(len(data_input) - 1):
        X = float(i)
        Y = float(data_input[i])
        index.append(X)
        data.append(Y)
    # 创建小波对象并定义参数:
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
    data_output = pywt.waverec(coeffs, 'db8')
    if len(data_output)!= len(data_input):
        data_output = np.append(data_output, data_output[len(data_output) - 1])
    return data_output
"""一些别的函数"""
def diff(input):
    """这里是对输入数据进行微分，所以这里的起始值肯定是零"""
    res = [0]
    for i in range(1,len(input)):
        res.append(input[i] - input[i-1])
    return np.array(res)

def linear_regression(x, y):
    """可以实现线性拟合"""
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)
