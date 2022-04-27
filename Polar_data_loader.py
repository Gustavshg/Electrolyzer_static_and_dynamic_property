import pandas
import numpy as np
from sklearn.model_selection import train_test_split

class Polar_Data_Loader():
    """通过一个类来读取数据，并且随机给到batch，主要是训练时使用的数据"""
    def __init__(self,SourceFile = '20s/Polar Fitting/All Polar-like data.csv'):
        df = pandas.read_csv(SourceFile)
        self.length = len(df)

        voltage = np.array(df['Voltage']).reshape(len(df), 1)
        current = np.array(df['Current']).reshape(len(df), 1)
        T_in = np.array(df['T_in']).reshape(len(df), 1)
        T_out = np.array(df['T_out']).reshape(len(df), 1)
        LyeFlow = np.array(df['LyeFlow']).reshape(len(df), 1)

        self.voltage = np.array(df['Voltage']).reshape(len(df), 1) / 34
        self.current_density = np.array(df['Current']).reshape(len(df), 1) / 0.425
        self.T_out = np.array(df['T_out']).reshape(len(df), 1)
        self.T_in = np.array(df['T_in']).reshape(len(df), 1)
        self.LyeFlow = np.array(df['LyeFlow']).reshape(len(df), 1)

        lam = 0.204425 / LyeFlow
        current /= 0.425 * 4000
        voltage /= 34
        T_out /= 100
        T_in /= 100

        self.all_data_x = np.concatenate((T_out, T_in, current, lam), axis=1)
        self.all_data_y = voltage

    def get_batch(self, batch_size=50):
        """按照批次给出神经网络训练所需要的数据"""
        seq = []
        vol = []
        for i in range(batch_size):
            index = np.random.randint(0, self.length)
            seq.append(self.all_data_x[index])
            vol.append(self.all_data_y[index])
        return np.array(seq), np.array(vol)

    def get_polar_data(self):
        '''这部分是针对普通极化曲线函数的输入需求，返回值，这部分的数据是没有经过标准化的'''
        return self.T_out, self.T_in, self.current_density, self.LyeFlow

    def get_voltage(self):
        """针对普通极化曲线给出真实电压值"""
        return self.voltage

    def get_all_data(self):
        """退回神经网络需要的所有数据"""
        return self.all_data_x, self.all_data_y

class Original_Data_Loader():
    """通过一个类来读取数据，并且随机给到batch，主要是复现和对比时的数据"""
    def __init__(self,SourceFile = '2s/Original/TJ-20210924.csv'):
        df = pandas.read_csv(SourceFile)
        # self.df = pandas.read_csv(SourceFile)
        OriginalColumns = ['时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
                           '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
                           'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
                           '进罐温度', '进罐压力']
        self.length = len(df)
        df['出口温度'] =( df['氢槽温'] + df['氧槽温']) / 2
        voltage = np.array(df['电解电压']).reshape(len(df), 1)
        current = np.array(df['电解电流']).reshape(len(df), 1)
        T_in = np.array(df['碱温']).reshape(len(df), 1)
        T_out = np.array(df['出口温度']).reshape(len(df), 1)
        LyeFlow = np.array(df['碱液流量']).reshape(len(df), 1)

        self.voltage = np.array(df['电解电压']).reshape(len(df), 1) / 34
        self.current_density = np.array(df['电解电流']).reshape(len(df), 1) / 0.425
        self.T_in = np.array(df['碱温']).reshape(len(df), 1)
        self.T_out = np.array(df['出口温度']).reshape(len(df), 1)
        self.LyeFlow = np.array(df['碱液流量']).reshape(len(df), 1)

        lam = 0.204425 / LyeFlow
        current /= 0.425 * 4000
        voltage /= 34
        T_out /= 100
        T_in /= 100

        self.all_data_x = np.concatenate((T_out, T_in, current, lam), axis=1)
        self.all_data_y = voltage

    def get_batch(self, batch_size=50):
        """按照批次给出神经网络训练所需要的数据"""
        seq = []
        vol = []
        for i in range(batch_size):
            index = np.random.randint(0, self.length)
            seq.append(self.all_data_x[index])
            vol.append(self.all_data_y[index])
        return np.array(seq), np.array(vol)

    def get_polar_data(self):
        '''这部分是针对普通极化曲线函数的输入需求，返回值，这部分的数据是没有经过标准化的'''
        return self.T_out, self.T_in, self.current_density, self.LyeFlow

    def get_voltage(self):
        """针对普通极化曲线给出真实电压值"""
        return self.voltage

    def get_all_data(self):
        """退回神经网络需要的所有数据"""
        return self.all_data_x, self.all_data_y






class Test_all_data_loader():
    def __init__(self,SourceFile = '20s/Polar Fitting/All Polar-like data.csv'):
        df = pandas.read_csv(SourceFile)
        self.length = len(df)
        voltage = np.array(df['Voltage']).reshape(len(df), 1)
        current = np.array(df['Current']).reshape(len(df), 1)
        T_in = np.array(df['T_in']).reshape(len(df), 1)
        T_out = np.array(df['T_out']).reshape(len(df), 1)
        LyeFlow = np.array(df['LyeFlow']).reshape(len(df), 1)

        self.voltage = np.array(df['Voltage']).reshape(len(df), 1) / 34
        self.current_density = np.array(df['Current']).reshape(len(df), 1) / 0.425
        self.T_out = np.array(df['T_out']).reshape(len(df), 1)
        self.T_in = np.array(df['T_in']).reshape(len(df), 1)
        self.LyeFlow = np.array(df['LyeFlow']).reshape(len(df), 1)

        lam = 0.204425 / LyeFlow
        current /= 0.425 * 4000
        voltage /= 34
        T_out /= 100
        T_in /= 100

        self.all_data_x = np.concatenate((T_out, T_in, current, lam), axis=1)
        self.all_data_y = voltage

        self.train_x, self.test_x,self.train_y,  self.test_y = train_test_split(self.all_data_x, self.all_data_y,
                                                                                test_size=0.20)

    def get_batch(self, batch_size=50):
        """按照批次给出神经网络训练所需要的数据，所有的训练数据都从这里来，包括非神经网咯的部分"""
        seq = []
        vol = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.train_x))
            seq.append(self.train_x[index])
            vol.append(self.train_y[index])
        return np.array(seq), np.array(vol)

    def get_test(self):
        return self.test_x, self.test_y

    def get_polar_data(self):
        """这部分是针对普通极化曲线函数的输入需求，返回值，这部分的数据是没有经过标准化的"""
        return self.T_out, self.T_in, self.current_density, self.LyeFlow

    def get_voltage(self):
        """针对普通极化曲线给出真实电压值"""
        return self.voltage

    def get_all_data(self):
        """退回神经网络需要的所有数据"""
        return self.all_data_x, self.all_data_y
