# e problem of the model in this version is listed in the 3rd section of https://lgkndmgws8.feishu.cn/docs/doccnj6oS07hvBIjSdoTcWoMhbe#olhTDh

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import scipy as sp
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random

def remove9999(originaldata):
    return  originaldata.drop(originaldata[originaldata.V<0].index)#deleting records where voltage is minus

t0 = time.time()
datapath = "AllData-20s.csv"
smoothdatapath = "SmoothData-20s.csv"

dataread = 1 # if 1, read written dataset; if 0 calculate dataset from scratch

if dataread ==0:
    originaldata = pd.read_csv(datapath)
    Dataset = remove9999(originaldata)# from this step, there are not non-sensable data in the dataframe
    Dataset['OriTemp'] = (Dataset.TO2+Dataset.TH2)/2
    Dataset['Power'] = Dataset.V*Dataset.I
    Dataset['HeatLyeIn'] = Dataset.Qlye * Dataset.Tlye
    Dataset['HeatLyeOut'] = Dataset.Qlye * Dataset.OriTemp
    RequiredSets = ['Power','OriTemp','HeatLyeIn','HeatLyeOut','V','I']#DataSets that are open to smoothening
    for item in RequiredSets:#Smoothening process is extremely time-consuming
        for i in np.arange(1,len(Dataset)-1):
            #Dataset[item][i] = np.average(Dataset[item][i-1:i+1])
            Dataset[item][i] = (Dataset[item][i-1]+Dataset[item][i]+Dataset[item][i+1])/3
    Dataset.to_csv("SmoothData-20s.csv")

if dataread ==1:
    Dataset = pd.read_csv(smoothdatapath)

#here linear regression
predictlength = 1
Dataset['PredTemp'] =Dataset['OriTemp'].shift(-predictlength)
Dataset['DeltaTemp'] = Dataset['PredTemp']-Dataset['OriTemp']
Dataset.dropna(inplace=True)
length = len(Dataset)

#concatenate input and output
inputcolumns = ['Power','OriTemp','HeatLyeIn','HeatLyeOut'] #columns required in the input region
outputcolumns = ['DeltaTemp'] #columns required in the output region
X = np.array(Dataset[inputcolumns[0]]).reshape(length,1)


for col in inputcolumns[1:]:
    X = np.concatenate((X,np.array(Dataset[col]).reshape(length,1)),axis =1)
for col in outputcolumns:
    y = np.array(Dataset[col]).reshape(length,1)

randomset = 1#if randomset ==1, shuffle input and output vector
X = preprocessing.scale(X)#X is for original input before shuffle, and predict after regression
X_input = X.copy()

y_input = y.copy()
if randomset == 1:
    random.seed(11224)
    random.shuffle(X_input)
    random.shuffle(y_input)


X_train,X_test,y_train,y_test = train_test_split(X_input,y_input,test_size=0.25)
clf = LinearRegression(n_jobs=-1)

clf.fit(X_train,y_train)
confidence = clf.score(X_test,y_test)
print(confidence)
Temp_predict = clf.predict(X)
'''
ReviveTemp=[]
for delta in np.arange(len(Dataset['DeltaTemp'])):
    ReviveTemp.append(Dataset['OriTemp']+sum(Dataset['DeltaTemp'][:delta]))
    print(delta)

plt.plot(ReviveTemp,label= 'predicted temp')
'''

print(len(Dataset))
print(len(Temp_predict))
Dataset['DeltaPredTemp'] = Temp_predict

Databuffer = 0
if Databuffer==1:#Need to write in buffer file to check data
    Dataset.to_csv("./20s/BufferData-20s.csv")


print(clf.coef_)
tend = time.time()
print('the total consumed time is ',tend-t0)
#plt.scatter(np.arange(len(Dataset)),Dataset['DeltaTemp'],label='original delta temp')
#plt.scatter(np.arange(len(Dataset)),Temp_predict,label='predicted \delta temp')




AllData_20s = pd.read_csv('AllData-20s.csv')
AllData_20s = remove9999(AllData_20s)
fig,ax = plt.subplots(1,1)
ax_sub = ax.twinx()
lv,=ax.plot(AllData_20s['V'],'r')
li,=ax_sub.plot(AllData_20s['I']/0.85,'g')


plt.legend(handles=[lv,li],labels = ['Voltage','Current density'],loc = 4)
plt.show()