"""在这文件中，我们需要读取李昊提取后的红外文件，并且进行分析，这个文件可能会集中考虑将txt转化成csv的问题"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas
from scipy import optimize
from sklearn.metrics import r2_score
plt.style.use('seaborn')

def read_infra(filename="Infra images/1126/IMG20211126101252.txt"):
    file = open(filename)
    contents = file.readlines()
    file.close()
    img = np.array([[0]] * 160)
    for line in contents:
        line = line.split()
        cur = list(map(float, line))
        cur = np.array(cur)
        cur = np.expand_dims(cur, axis=1)
        img = np.concatenate((img, cur), axis=1)

    img = img[:, 1:]
    img = np.fliplr(img)
    img = np.flipud(img)
    return img


def temp_distribution(img=read_infra()):
    n = 12  # 要将图像如何进行分段（纵向分段）
    step = len(img) // n

    seq = []
    df = pandas.DataFrame(columns=['loc', 'temp'])
    data = np.zeros([1560, 1])
    for i in range(n):
        cur = img[i * step:(i + 1) * step, :]
        cur = cur.flatten()
        cur = np.expand_dims(cur, axis=1)
        data = np.concatenate((data, cur), axis=1)

    data = data[:, 1:]
    return data.T
    # plt.figure(figsize=(15, 8))
    # plt.title('temp distribution over entire img')
    # plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    # seaborn.violinplot(scale='count', data=data, orient='h')
    # plt.legend()
    # plt.show()


def list2hist(listdata):
    # 这里需要将列表数据转化为一个histogram的数据
    mini = min(listdata)
    maxi = max(listdata)
    step = 100
    x = np.linspace(start=mini, stop=maxi, num=step)
    y = np.zeros(step)
    for n in listdata:
        for i in range(1, step):
            if n < x[i]:
                y[i - 1] += 1
                break
            elif n == x[i]:
                y[i] += 1
                break
    return x, y


def gaussian(x, height, center, width, offset):
    return height * np.exp(-(x - center) ** 2 / (2 * width ** 2)) + offset


def three_gaussians(x,
                    h1, c1, w1,
                    h2, c2, w2,
                    h3, c3, w3,
                    offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
            gaussian(x, h2, c2, w2, offset=0) +
            gaussian(x, h3, c3, w3, offset=0) +
            offset)


def two_gaussians(x,
                  h1, c1, w1,
                  h2, c2, w2,
                  offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
            gaussian(x, h2, c2, w2, offset=0) +
            offset)


def gaussians_fitting(x, height):
    '''数据的极限'''
    low = x[0]
    high = x[-1]
    '''误差方程'''
    errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y) ** 2
                               # + p[0] ** 2 + p[3] ** 2 + p[-1] **4
    errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y) ** 2
                               # + p[0] ** 2 + p[3] ** 2 + p[6] ** 2  + p[-1] **4#参数的正则化惩罚项
    '''初始化参数'''
    guess3 = [20.22, low * 0.75 + high * 0.25, 10.,
              30.50, (low + high)/2, 10.,
              30.50, low * 0.25 + high * 0.75, 10., 0]
    param_bound_3 = ([5.  ,low,1.  ,5.  ,low,1.  ,5.  ,low,1.  ,-5.],
                     [200.,high,200.,200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制

    guess2 = [10.22,low * 0.6 + high * 0.4, 10.,
              50.50, low * 0.4 + high * 0.6, 10., 0]
    param_bound_2 = ([5.  ,low,1.  ,5.  ,low,1.  ,-5.],
                     [200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制
    '''首先进行三曲线拟合'''
    optim3, ssss = optimize.curve_fit(three_gaussians, x, height, p0=guess3, bounds=param_bound_3)

    '''如果三曲线拟合结果中，有两个分布均值接近，则删减一条'''
    if min(abs(optim3[1] - optim3[4]),abs(optim3[7] - optim3[4]),abs(optim3[1] - optim3[7])) > 5:
        optim = optim3
    else:
        optim2, ssss = optimize.curve_fit(two_gaussians, x, height, p0=guess2, bounds=param_bound_2)
        optim = optim2
    return optim


def temp_slope_analysis(temp_distribution_data):#数据输入应当是temp_distribution函数的结果
    i = 0
    seq = []#显示位置的序列，主要是当前区域在真实温度分布中从上到下的次序
    temp_big = []#主要峰值的温度
    temp_small = []#次要峰值的温度
    height_big = []#主要峰值的高度，这里是出现的次数，在高斯取样时需要进行这么多次的取样
    height_small = []#次要峰值的高度，这里是出现的次数，在高斯取样时需要进行这么多次的取样
    width_big = []#主要峰的方差
    width_small = []#次要峰的方差
    for line in temp_distribution_data:
        if not i in [0,8,11]:
            x, height = list2hist(line)
            res = gaussians_fitting(x, height)
            seq.append(i*1.0)
            for k in [1,4,7]:
                if k<len(res)-1 and res[k]<30:#如果峰值温度小于30摄氏度，就忽略这个峰值
                    res[k-1] = 0
            if len(res) == 10:
                temps = np.array([res[1],res[4],res[7]])
                heights = np.array([res[0],res[3],res[6]])
                widths = np.array([res[2],res[5],res[8]])
                order = np.argsort(heights)
                temp_big.append(temps[order[-1]])
                temp_small.append(temps[order[-2]])
                height_big.append(heights[order[-1]])
                height_small.append(heights[order[-2]])
                width_big.append(widths[order[-1]])
                width_small.append(widths[order-2])
            else:
                temps = np.array([res[1],res[4]])
                heights = np.array([res[0],res[3]])
                widths = np.array([res[2], res[5]])
                order = np.argsort(heights)
                temp_big.append(temps[order[-1]])
                temp_small.append(temps[order[-2]])
                height_big.append(heights[order[-1]])
                height_small.append(heights[order[-2]])
                width_big.append(widths[order[-1]])
                width_small.append(widths[order-2])
        i += 1
    slope, intercept = np.polyfit(seq, temp_big, 1)#这里是对以上的结果进行斜率分析,SLOPE是斜率, INTERCEPT是截距，这里的方法比较原始，没有对正态分布进行取样
    print('original slope: %f, original intercept: %f'%(slope,intercept))
    '''以下是通过正态取样方法对主要峰进行分析，最后得到slope和intercept'''

    x_seq = []
    y_seq = []
    for i in range(len(seq)):
        cur = np.random.normal(loc = temp_big[i],scale = width_big[i],size = int(height_big[i]))
        for j in range(len(cur)):
            x_seq.append(seq[i])
            y_seq.append(cur[j])
    slope, intercept = np.polyfit(x_seq, y_seq, 1)#这里是对以上的结果进行斜率分析,SLOPE是斜率, INTERCEPT是截距，这里的方法比较原始，没有对正态分布进行取样
    print('normal sampled slope: %f, normal sampled intercept: %f'%(slope,intercept))
    return seq,temp_big,temp_small,height_big,height_small,width_big,width_small,slope,intercept

data = temp_distribution()


compare = 1
if compare == 1:

    plt.figure(figsize=(15, 8))
    plt.title('Distribution in temperature')
    plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    plt.xlabel('From top to bottom')
    plt.ylabel(r'$Temperature \ (^\circ C)$')
    '''原始数据的小提琴图'''
    hh = 1
    x = np.array([0])
    y = np.array([0])
    for line in data:

        hhh = [hh] * len(line)
        x = np.concatenate((x, hhh))
        y = np.concatenate((y, line))
        hh += 1

    x = x[1:]
    y = y[1:]
    seaborn.violinplot(x, y)

    # plt.figure(figsize=(15, 8))
    # plt.title('Distribution in temperature')
    # plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    # plt.xlabel('From top to bottom')
    # i = 1
    '''正态多峰分析的部分'''
    i = 0

    for line in data:
        if not i in [0,8,11]:
            x, height = list2hist(line)
            res = gaussians_fitting(x, height)
            for j in [0,3,6]:
                if j <len(res) -1:
                    res[j] *= 10
            if len(res) == 10:
                plt.scatter(i, res[1], res[0], edgecolors='b')
                plt.scatter(i, res[4], res[3], edgecolors='b')
                plt.scatter(i, res[7], res[6], edgecolors='b')
            else:
                plt.scatter(i, res[1], res[0], edgecolors='b')
                plt.scatter(i, res[4], res[3], edgecolors='b')
        i += 1
    #需要将上面的seq、temp、height、width数组保存成Dataframe


    # plt.figure(figsize=(15, 8))#如果需要单独展示，就取消注释
    # plt.title('Distribution in temperature')
    # plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    # plt.xlabel('From top to bottom')
    # plt.ylabel(r'$Temperature \ (^\circ C)$')

    seq, temp_big, temp_small, height_big, height_small, width_big, width_small,slope,intercept = temp_slope_analysis(data)




    plt.plot(seq,temp_big,label = 'Primary center')
    plt.plot(seq,temp_small,label = 'Secondary center')
    '''线性拟合与分析'''

    plt.plot([0,11],[intercept,intercept+12*slope],'red',label = 'Primary fitted, slope = %f'%(slope))

    plt.legend()
    plt.show()



exam = 0
if exam == 1:
    '''单独抽取出一根来进行分解，只用作有输出出错时使用'''
    x,height = list2hist(data[3,:])
    low = x[0]
    high = x[-1]
    errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y) ** 2
                               # + p[0] ** 2 + p[3] ** 2 + p[-1] **4
    errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y) ** 2
                               # + p[0] ** 2 + p[3] ** 2 + p[6] ** 2  + p[-1] **4#参数的正则化惩罚项

    guess3 = [20.22, low + 2, 10.,
              30.50, (low + high)/2, 10.,
              30.50, high - 2, 10., 0]
    param_bound_3 = ([5.  ,low,1.  ,5.  ,low,1.  ,5.  ,low,1.  ,-5.],
                     [200.,high,200.,200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制

    guess2 = [10.22, low + 2, 10.,
              50.50, high - 2, 10., 0]
    param_bound_2 = ([5.  ,low,1.  ,5.  ,low,1.  ,-5.],
                     [200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制

    params, ssss = optimize.curve_fit(three_gaussians, x, height, p0=guess3, bounds=param_bound_3)
    optim3, success = optimize.leastsq(errfunc3, params, args=(x, height))#需去掉这部分

    if min(abs(optim3[1] - optim3[4]),abs(optim3[7] - optim3[4]),abs(optim3[1] - optim3[7])) > 5:
        optim = optim3
        print('center temperature:' + str(optim[1]))
        print('center temperature:' + str(optim[4]))
        print('center temperature:' + str(optim[7]))
        print(optim, success)

        plt.figure(figsize=(15, 8))
        plt.title('3 gaussian fitting')
        plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
        plt.scatter(x, height, marker='o', edgecolors=None)
        plt.plot(x, three_gaussians(x, *optim))
        plt.legend(['original', 'gaussian fitted'])

    else:
        params, ssss = optimize.curve_fit(two_gaussians, x, height, p0=guess2, bounds=param_bound_2)
        optim2, success = optimize.leastsq(errfunc2, params, args=(x, height))#需去掉这部分

        optim = optim2
        print('center temperature:' + str(optim[1]))
        print('center temperature:' + str(optim[4]))
        print(optim, success)

        plt.figure(figsize=(15, 8))
        plt.title('2 gaussian fitting')
        plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
        plt.scatter(x, height, marker='o', edgecolors=None)
        plt.plot(x, two_gaussians(x, *optim))
        plt.legend(['original', 'gaussian fitted'])

plt.show()