"""这里主要是开展截距与斜率之间对应关系的研究"""
import pandas
import numpy as np
import matplotlib.pyplot as plt
import Smoothen as sm
# 引入matplotlib字體管理 FontProperties
from matplotlib.font_manager import FontProperties
# 設置我們需要用到的中文字體（字體文件地址）
my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)
plt.style.use('seaborn')

source_file = 'Infra images/Analysis result/slope and intercept 1202'
df = pandas.read_csv(source_file)
print(len(df))

slopes = df['slope']
intercepts = df['intercept']


plt.figure(figsize=(15, 8))
plt.xlabel('intercept temperature')
plt.ylabel('corresponding slope')
plt.title('1201电解槽温度分布截距与对应温度分布斜率',fontproperties=my_font)
s = 0
e = 190
a0 , a1 = sm.linear_regression(intercepts[s:e],slopes[s:e])
_X1 = [min(intercepts[s:e]),max(intercepts[s:e])]
_Y1 = [a0 + a1 * x for x in _X1]
plt.plot(_X1,_Y1)
plt.scatter(intercepts[s:e],slopes[s:e],alpha=0.3,label = 'start up with  linear slope %f, intercept %f'%(a1,a0))


s = e
e = 420
a0 , a1 = sm.linear_regression(intercepts[s:e],slopes[s:e])
_X1 = [min(intercepts[s:e]),max(intercepts[s:e])]
_Y1 = [a0 + a1 * x for x in _X1]
plt.plot(_X1,_Y1)
plt.scatter(intercepts[s:e],slopes[s:e],alpha=0.3,label = '1st polar with linear slope %f, intercept %f'%(a1,a0))

s = e
e = 520
a0 , a1 = sm.linear_regression(intercepts[s:e],slopes[s:e])
_X1 = [min(intercepts[s:e]),max(intercepts[s:e])]
_Y1 = [a0 + a1 * x for x in _X1]
plt.plot(_X1,_Y1)
plt.scatter(intercepts[s:e],slopes[s:e],alpha=0.3,label = 'rapid heating with linear slope %f, intercept %f'%(a1,a0))

s = e
e = 820
a0 , a1 = sm.linear_regression(intercepts[s:e],slopes[s:e])
_X1 = [min(intercepts[s:e]),max(intercepts[s:e])]
_Y1 = [a0 + a1 * x for x in _X1]
plt.plot(_X1,_Y1)
plt.scatter(intercepts[s:e],slopes[s:e],alpha=0.3,label = '2nd polar with linear slope %f, intercept %f'%(a1,a0))

s = e
e = 1150
a0 , a1 = sm.linear_regression(intercepts[s:e],slopes[s:e])
_X1 = [min(intercepts[s:e]),max(intercepts[s:e])]
_Y1 = [a0 + a1 * x for x in _X1]
plt.plot(_X1,_Y1)
plt.scatter(intercepts[s:e],slopes[s:e],alpha=0.3,label = '3rd polar with linear slope %f, intercept %f'%(a1,a0))

s = e
e = len(slopes)
a0 , a1 = sm.linear_regression(intercepts[s:e],slopes[s:e])
_X1 = [min(intercepts[s:e]),max(intercepts[s:e])]
_Y1 = [a0 + a1 * x for x in _X1]
plt.plot(_X1,_Y1)
plt.scatter(intercepts[s:e],slopes[s:e],alpha=0.3,label = 'cooling down with linear slope %f, intercept %f'%(a1,a0))
plt.legend()
plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
# plt.show()