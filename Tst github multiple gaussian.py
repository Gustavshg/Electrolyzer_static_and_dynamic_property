import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd


#--------get data--------------------

# file = open('Cache/multiple_gaussian_fit_test.txt')
# contents = file.readlines()
# file.close()
# print(len(contents))
# data = np.array([[0]] * 3)
# for line in contents:
# 	line = line.split(',')
# 	cur = list(map(float,line))
# 	cur = np.array(cur)
# 	cur = np.expand_dims(cur,axis=1)
# 	data = np.concatenate((data,cur),axis=1)
#
# data = data[:,1:]
# data = data.T
# data = pd.DataFrame(data)
# data.reset_index(drop=True)
# data.to_csv('Cache/multiple_gaussian_fit_test.csv')
# print(data)
# print(data.shape)

#------------------------------------

data = pd.read_csv('Cache/multiple_gaussian_fit_test.csv')
data = np.array(data)
data = data[:,1:]
print(data)



def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def six_gaussians(x, h1, c1, w1,
		h2, c2, w2,
		h3, c3, w3,
		h4, c4, w4,
		h5, c5, w5,
		h6, c6, w6,
		offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
        gaussian(x, h2, c2, w2, offset=0) +
        gaussian(x, h3, c3, w3, offset=0) +
        gaussian(x, h4, c4, w4, offset=0) +
        gaussian(x, h5, c5, w5, offset=0) +
        gaussian(x, h6, c6, w6, offset=0) +
        offset)

errfunc6 = lambda p, x, y: (six_gaussians(x, *p) - y)**2

guess6= [.22, 360, 65,
	.22, 834, 65,
	.39, 1164, 140,
	.59, 1550, 200,
	.3, 1990, 200,
	.3, 2350, 75, 0]
optim6, success = optimize.leastsq(errfunc6, guess6[:], args=(data[:,0], data[:,2]))

print("Fundamental frequency:  ampl ={0:5.1f}, freq ={1:5.1f} Hz, sigma ={2:5.2f} Hz".format(optim6[0],optim6[1],optim6[2]/2))
print( "First harmonic:  ampl ={0:5.1f}, freq ={1:5.1f} Hz, sigma ={2:5.2f} Hz".format(optim6[3],optim6[4],optim6[5]/2))
print( "Second harmonic:  ampl ={0:5.1f}, freq ={1:5.1f} Hz, sigma ={2:5.2f} Hz".format(optim6[6],optim6[7],optim6[8]/2))
print( "Third harmonic:  ampl ={0:5.1f}, freq ={1:5.1f} Hz, sigma ={2:5.2f} Hz".format(optim6[9],optim6[10],optim6[11]/2))
print( "Fourth harmonic:  ampl ={0:5.1f}, freq ={1:5.1f} Hz, sigma ={2:5.2f} Hz".format(optim6[12],optim6[13],optim6[14]/2))
print( "Fifth harmonic:  ampl ={0:5.1f}, freq ={1:5.1f} Hz, sigma ={2:5.2f} Hz".format(optim6[15],optim6[16],optim6[17]/2))



plt.scatter(data[:,0], data[:,2], c='pink', label='measurement', marker='.', edgecolors=None)
plt.plot(data[:,0], six_gaussians(data[:,0], *optim6),
    c='b', label='fit of 6 Gaussians')
plt.title("FFT of white noise hitting an open metal tube")
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.legend(loc='upper left')
plt.savefig('result.png')
plt.show()