import numpy
import numpy as np
import pandas as np
import pandas
import matplotlib.pyplot as plt
import pandas as pd

import time
import os
SelectedFiles = ["20s/Original/TJ-20211126.csv"]
OriginalColumns = [ '时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
       '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
       'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
       '进罐温度', '进罐压力']
df = pandas.read_csv(SelectedFiles[0])
df['T_out'] = ( df['氧槽温'] + df['氢槽温'])/2
df['T_in'] = df['碱温']
df['Current'] = df['电解电流']
df['Voltage'] = df['电解电压']

T_in = np.array(df['T_in'])
T_out = np.array(df['T_out'])
current = np.array(df['Current'])
voltage = np.array(df['Voltage'])
data = np.array([T_in,T_out])
newdf = pandas.DataFrame()
newdf['T_in'] = T_in
newdf['T_out'] = T_out
newdf['current'] = current
newdf['voltage'] = voltage
tofiledf = pandas.DataFrame()
plt.plot(T_out)
tofiledf = newdf[2200:2300]
#tofiledf = tofiledf.append(newdf[4000:4100])
#tofiledf = tofiledf.append(newdf[400:450])
#tofiledf = tofiledf.append(newdf[510:560])
#tofiledf = tofiledf.append(newdf[620:670])
#tofiledf = tofiledf.append(newdf[715:755])
#tofiledf = tofiledf.append(newdf[800:840])
#tofiledf = tofiledf.append(newdf[920:970])
#tofiledf = tofiledf.append(newdf[1100:1200])
#tofiledf = tofiledf.append(newdf[1250:1300])
#tofiledf.to_csv('20s/1130 Polar data.csv')
print(tofiledf)
plt.show()