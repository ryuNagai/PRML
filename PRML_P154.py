# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:49:57 2018

@author: Ryutaro
"""

import random, numpy as np
import matplotlib.pyplot as plt

#make N order polynominal
def polynominal(x,w): #w is list of parameter
    y = 0
    for i in range(len(w)):
        y += w[i] * x**i
    return y


def makeDataAndSample(bias,w1,var):
    x_size = 100 #データ点の数
    x = np.zeros(x_size)
    t = np.zeros(x_size)
    #bias = -0.3
    #w1 = 0.5
    a = [bias, w1] #求めたい真のパラメータ
    for i in range(0,x_size):
        x[i] = random.uniform(-1.0,1.0) #データ点をランダムに生成
        noise = random.gauss(0,var) #ガウスノイズ
        t[i] = polynominal(x[i],a) + noise 
        #真のパラメータから生成される多項式にガウスノイズを加える
    
    return x,t
    
def D2Gaussian(X,Y,mu,var):
    det = np.linalg.det(var) #det(var)
    inv_var = np.linalg.inv(var) #det( inv(var) )
    #X, Y = np.meshgrid(x, y)
    
    def f(x,y):
        x_c = np.array([x, y]) - mu #mean point
        f = np.exp(- x_c.dot(inv_var).dot(x_c[np.newaxis, :].T) / 2.0) / (2*np.pi*np.sqrt(det))
        return f
    
    Z = np.vectorize(f)(X,Y) 
    return Z

def makePrior(beta,mu,BIAS,W1):
    #bias = w1 = np.arange(-1, 1, .01)
    var = np.array([[1/beta,0],[0,1/beta]])
    prior = D2Gaussian(BIAS,W1,mu,var)
    '''
    if __name__ == '__main__':
        z_min, z_max = -np.abs(prior).max(), np.abs(prior).max()
        #W1, W2 = np.meshgrid(bias, w1)
        plt.pcolor(BIAS, W1, prior, cmap='RdBu', vmin=z_min, vmax=z_max)
        plt.title('prior of w')
        plt.xlabel('bias')
        plt.ylabel('w1')
        # set the limits of the plot to the limits of the data
        plt.axis([BIAS.min(), BIAS.max(), W1.min(), W1.max()])
        plt.colorbar()
    '''
    return prior

def likelifoodFunc(t,x,bias,w1,var):
    beta = 1/var #accuracy parameter
    mu = polynominal(x,[bias,w1]) 
    f =  np.sqrt(beta / (2 * np.pi) ) * np.exp(-(t - mu) ** 2 * (beta / 2) )
    return f

def likelifoodDist(target,data,BIAS,W1,var):
    #w1 = w2 = np.arange(-1, 1, .01)
    LF = np.vectorize(likelifoodFunc)(target,data,BIAS,W1,var)
    '''
    if __name__ == '__main__':
        z_min, z_max = -np.abs(LF).max(), np.abs(LF).max()
        plt.pcolor(BIAS, W1, LF, cmap='RdBu', vmin=z_min, vmax=z_max)
        plt.title('likelifood of t')
        plt.xlabel('bias')
        plt.ylabel('w1')
        # set the limits of the plot to the limits of the data
        plt.axis([BIAS.min(), BIAS.max(), W1.min(), W1.max()])
        plt.colorbar()
    '''
    return LF

def posterior(BIAS,W1,prior,LF):
    POST = prior * LF
    normalizedPOST = POST / np.sum(POST)
    '''
    if __name__ == '__main__':
        z_min, z_max = -np.abs(POST).max(), np.abs(POST).max()
        plt.pcolor(BIAS, W1, POST, cmap='RdBu', vmin=z_min, vmax=z_max)
        plt.title('posterior of w')
        plt.xlabel('bias')
        plt.ylabel('w1')
        # set the limits of the plot to the limits of the data
        plt.axis([BIAS.min(), BIAS.max(), W1.min(), W1.max()])
        plt.colorbar()
    ''' 
    return normalizedPOST

#ここからmain
#真のパラメータ
bias = -0.3
w1 = 0.5
var = 0.2 #これだけ最初からわかってるものとする;Hyper parameter
beta = 1 / var

#観測点と観測データを用意
data,target = makeDataAndSample(bias,w1,var)

#parameter空間を定義
bias = w1 = np.arange(-1, 1, 0.005)
BIAS, W1 = np.meshgrid(bias, w1)

prior = makePrior(2,0,BIAS,W1)

for i in range(0,len(data)):

    LF = likelifoodDist(target[i],data[i],BIAS,W1,var)
    normalizedPOST = posterior(BIAS,W1,prior,LF)
    
    if i % 10 == 0:
        plt.figure(i)
        z_min, z_max = -np.abs(normalizedPOST).max(), np.abs(normalizedPOST).max()
        plt.pcolor(BIAS, W1, normalizedPOST, cmap='RdBu', vmin=z_min, vmax=z_max)
        plt.title('posterior of w')
        plt.xlabel('bias')
        plt.ylabel('w1')
        # set the limits of the plot to the limits of the data
        plt.axis([BIAS.min(), BIAS.max(), W1.min(), W1.max()])
        plt.colorbar()
    
    prior = normalizedPOST

