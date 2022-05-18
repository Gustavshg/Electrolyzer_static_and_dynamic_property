import numpy as np
import pandas
from scipy import optimize


def read_infra(filename):
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

def img_show(img,file):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.title(
        'Distribution in temperature of: ' + file[-14:-10] + ' ' + file[-10:-8] + ':' + file[-8:-6] + ':' + file[-6:-4])
    plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    plt.imshow(img)
    plt.colorbar()

def temp_distribution(img):
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
    # guess3 = [20.22, low * 0.8 + high * 0.2, 10.,
    #           30.50, (low + high)/2, 10.,
    #           30.50, low * 0.2 + high * 0.8, 10., 0]
    guess3 = [20.22, low + 2, 10.,
              30.50, (low + high)/2, 10.,
              30.50, high - 2, 10., 0]
    param_bound_3 = ([5.  ,low,1.  ,5.  ,low,1.  ,5.  ,low,1.  ,-5.],
                     [200.,high,200.,200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制

    # guess2 = [10.22,low * 0.6 + high * 0.4, 10.,
    #           50.50, low * 0.4 + high * 0.6, 10., 0]
    guess2 = [10.22, low + 2, 10.,
              50.50, high - 2, 10., 0]
    param_bound_2 = ([5.  ,low,1.  ,5.  ,low,1.  ,-5.],
                     [200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制
    '''首先进行三曲线拟合'''
    optim3, ssss = optimize.curve_fit(three_gaussians, x, height, p0=guess3, bounds=param_bound_3,maxfev=50000)

    '''如果三曲线拟合结果中，有两个分布均值接近，则删减一条'''
    if min(abs(optim3[1] - optim3[4]),abs(optim3[7] - optim3[4]),abs(optim3[1] - optim3[7])) > 5:
        optim = optim3
    else:
        optim2, ssss = optimize.curve_fit(two_gaussians, x, height, p0=guess2, bounds=param_bound_2,maxfev=50000)
        optim = optim2
    return optim

def temp_slope_analysis(temp_distribution_data,threshold = 30,bars = [0,8,11]):#数据输入应当是temp_distribution函数的结果
    i = 0
    seq = []#显示位置的序列，主要是当前区域在真实温度分布中从上到下的次序
    temp_big = []#主要峰值的温度
    temp_small = []#次要峰值的温度
    height_big = []#主要峰值的高度，这里是出现的次数，在高斯取样时需要进行这么多次的取样
    height_small = []#次要峰值的高度，这里是出现的次数，在高斯取样时需要进行这么多次的取样
    width_big = []#主要峰的方差
    width_small = []#次要峰的方差
    for line in temp_distribution_data:
        if not i in bars:
            x, height = list2hist(line)
            res = gaussians_fitting(x, height)
            seq.append(i*1.0)
            for k in [1,4,7]:
                if k<len(res)-1 and res[k]<threshold:#如果峰值温度小于30摄氏度，就忽略这个峰值
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
    # slope, intercept = np.polyfit(seq, temp_big, 1)#这里是对以上的结果进行斜率分析,SLOPE是斜率, INTERCEPT是截距，这里的方法比较原始，没有对正态分布进行取样
    # print('original slope: %f, original intercept: %f'%(slope,intercept))
    '''以下是通过正态取样方法对主要峰进行分析，最后得到slope和intercept'''

    x_seq = []
    y_seq = []

    for i in range(len(seq)):
        cur = np.random.normal(loc = temp_big[i],scale = width_big[i],size = int(height_big[i])*10)
        for j in range(len(cur)):
            x_seq.append(seq[i])
            y_seq.append(cur[j])
    slope, intercept = np.polyfit(x_seq, y_seq, 1)#这里是对以上的结果进行斜率分析,SLOPE是斜率, INTERCEPT是截距，这里的方法比较原始，没有对正态分布进行取样
    # print('normal sampled slope: %f, normal sampled intercept: %f'%(slope,intercept))
    return seq,temp_big,temp_small,height_big,height_small,width_big,width_small,slope,intercept



def show_violin_gaussian_slope(data,filename,bars = [0,8,11]):
    #filename sample: 'Infra images/1125/IMG20211125093106.txt'
    import seaborn
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 8))
    plt.title('Distribution in temperature of: ' + filename[-14:-10]+' '+filename[-10:-8] + ':'+filename[-8:-6]+':'+filename[-6:-4])
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
    i = 0
    for line in data:
        if not i in bars:

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
    temp_min = min(data[2])
    temp_max = max(data[2])
    thre = 0.6 * temp_min + 0.4 * temp_max
    thre_max = 50
    if thre >= thre_max:
        thre = thre_max
    seq, temp_big, temp_small, height_big, height_small, width_big, width_small,slope,intercept = temp_slope_analysis(data,thre, bars)

    plt.plot(seq,temp_big,label = 'Primary center')
    plt.plot(seq,temp_small,label = 'Secondary center')
    '''线性拟合与分析'''

    plt.plot([0,11],[intercept,intercept+12*slope],'red',label = 'Primary fitted, slope = %f, intercept = %f'%(slope,intercept))

    plt.legend()

def exam(data,slice,filename):
    #filename sample: 'Infra images/1125/IMG20211125093106.txt'
    import seaborn
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    x,height = list2hist(data[slice,:])
    low = x[0]
    high = x[-1]
    errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y) ** 2
                               # + p[0] ** 2 + p[3] ** 2 + p[-1] **4
    errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y) ** 2
                               # + p[0] ** 2 + p[3] ** 2 + p[6] ** 2  + p[-1] **4#参数的正则化惩罚项
    '''原始的exam代码'''
    guess3 = [20.22, low + 2, 10.,
              30.50, (low + high)/2, 10.,
              30.50, high - 2, 10., 0]
    param_bound_3 = ([5.  ,low,1.  ,5.  ,low,1.  ,5.  ,low,1.  ,-5.],
                     [200.,high,200.,200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制

    # guess2 = [10.22, low + 2, 10.,
    #           50.50, high - 2, 10., 0]
    param_bound_2 = ([5.  ,low,1.  ,5.  ,low,1.  ,-5.],
                     [200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制
    print(guess3)
    '''使用了最终版slope分析代码'''
    # guess3 = [20.22, low * 0.8 + high * 0.2, 10.,
    #           30.50, (low + high)/2, 10.,
    #           30.50, low * 0.2 + high * 0.8, 10., 0]
    # param_bound_3 = ([5.  ,low,1.  ,5.  ,low,1.  ,5.  ,low,1.  ,-5.],
    #                  [200.,high,200.,200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制
    #
    guess2 = [10.22,low * 0.6 + high * 0.4, 10.,
              50.50, low * 0.4 + high * 0.6, 10., 0]
    # param_bound_2 = ([5.  ,low,1.  ,5.  ,low,1.  ,-5.],
    #                  [200.,high,200.,200.,high,200.,15.])#对每个参数的范围进行限制
    print(guess3)

    optim3, success = optimize.curve_fit(three_gaussians, x, height, p0=guess3, bounds=param_bound_3,maxfev=50000)
    # optim3, success = optimize.leastsq(errfunc3, params, args=(x, height))#需去掉这部分

    if min(abs(optim3[1] - optim3[4]),abs(optim3[7] - optim3[4]),abs(optim3[1] - optim3[7])) > 5:
        optim = optim3
        print('center temperature:' + str(optim[1]))
        print('center temperature:' + str(optim[4]))
        print('center temperature:' + str(optim[7]))
        # print(optim, success)

        plt.figure(figsize=(15, 8))
        plt.title('3 gaussian fitting of slice' +slice + 'in:'+ filename[-14:-10]+' '+filename[-10:-8] + ':'+filename[-8:-6]+':'+filename[-6:-4])
        plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
        plt.scatter(x, height, marker='o', edgecolors=None)
        plt.plot(x, three_gaussians(x, *optim))
        plt.legend(['original', 'gaussian fitted'])

    else:
        optim2, success = optimize.curve_fit(two_gaussians, x, height, p0=guess2, bounds=param_bound_2,maxfev=50000)
        # optim2, success = optimize.leastsq(errfunc2, params, args=(x, height))#需去掉这部分

        optim = optim2
        print('center temperature:' + str(optim[1]))
        print('center temperature:' + str(optim[4]))
        # print(optim, success)

        plt.figure(figsize=(15, 8))
        plt.title('2 gaussian fitting of slice ' +str(slice) + ' in:'+ filename[-14:-10]+' '+filename[-10:-8] + ':'+filename[-8:-6]+':'+filename[-6:-4])
        plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
        plt.scatter(x, height, marker='o', edgecolors=None)
        plt.plot(x, two_gaussians(x, *optim))
        plt.legend(['original', 'gaussian fitted'])
