# -*- coding: utf-8 -*-
"""
数据增强
"""
import numpy as np
from random import randint



def mix_up(x,y,s,alpha,n):
    x2=np.ones((n,32,32,32,1))
    y2=np.ones((n,2),dtype='float')
    s2= np.ones((n,32,32,32,1))
    ori_num=len(y)
    for i in range(n):
        index1=randint(0,ori_num-1)
        index2=randint(0,ori_num-1)
        b = np.random.beta(alpha,alpha)
        x2[i] = b*x[index1]+(1-b)*x[index2]
        y2[i] = b*y[index1]+(1-b)*y[index2] 
        s2[i]=b*s[index1]+(1-b)*s[index2] 
    return x2,y2,s2

def flip_dim1(x,y,s):
    n=len(y)
    x2=x
    y2=y
    s2=s
    for i in range(n):
        for j in range(16):
            tmp=x2[i,j,:,:,0].copy()
            x2[i,j,:,:,0]=x2[i,32-j-1,:,:,0]
            x2[i,32-j-1,:,:,0]=tmp    
            tmp=s2[i,j,:,:,0].copy()
            s2[i,j,:,:,0]=s2[i,32-j-1,:,:,0]
            s2[i,32-j-1,:,:,0]=tmp   
    return x2,y2,s2

def flip_dim2(x,y,s):
    n=len(y)
    x2=x
    y2=y
    s2=s
    for i in range(n):
        for j in range(16):
            tmp=x2[i,:,j,:,0].copy()
            x2[i,:,j,:,0]=x2[i,:,32-j-1,:,0]
            x2[i,:,32-j-1,:,0]=tmp    
            tmp=s2[i,:,j,:,0].copy()
            s2[i,:,j,:,0]=s2[i,:,32-j-1,:,0]
            s2[i,:,32-j-1,:,0]=tmp 
    return x2,y2,s2

def flip_dim3(x,y,s):
    n=len(y)
    x2=x
    y2=y
    s2=s
    for i in range(n):
        for j in range(16):
            tmp=x2[i,:,:,j,0].copy()
            x2[i,:,:,j,0]=x2[i,:,:,32-j-1,0]
            x2[i,:,:,32-j-1,0]=tmp  
            tmp=s2[i,:,:,j,0].copy()
            s2[i,:,:,j,0]=s2[i,:,:,32-j-1,0]
            s2[i,:,:,32-j-1,0]=tmp  
    return x2,y2,s2

