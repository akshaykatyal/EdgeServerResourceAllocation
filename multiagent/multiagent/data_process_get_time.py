#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:01:28 2019

@author: eva
"""
import numpy as np
import json
import matplotlib.pyplot as plt

dic= '/Users/tyuan/Documents/GitHub/DATA/2014-10/'

dic2= '/Users/tyuan/Documents/GitHub/DATA/SDNdata_24/'

offsize_lat = 22.9427
offsize_lng = 43.3175
zom = 10.256
size_rec = 100



def remove_seconds_improve(name):
# merger in 1 min
    fname = dic +'2014-10-'+name+'.txt'
    time = None
    with open(fname, 'r', encoding='ascii') as f:
        i =0
        sub_dic={}
        time=0
        vehicle_location=[]
        for j in f.readlines():
            se = j[:-1].split(',')
            if se[1][-5:-3] not in ['00','05','07']:
#                ,'30','35','37']:
                continue
            if time == 0:
                time = se[1][:-3]
            if time != se[1][:-3]:
                sub_dic[i]=vehicle_location
                vehicle_location=[]
                i+=1
                time = se[1][:-3]
                
                       
            if  float(se[4])>-22.9427 and float(se[4])< -22.8529 and float(se[5])< -43.2200 and float(se[5])> -43.3175:
                    x = (float(se[4])+offsize_lat)*zom*size_rec
                    y = (float(se[5])+offsize_lng)*zom*size_rec
                    load = np.random.uniform(10, 20)
                    item = [x,y,load]
                    vehicle_location.append(item)
                
        new_dic = {}
        k=0
        for i in range(0,len(sub_dic)-2,3):
            if len(sub_dic[i])>max(len(sub_dic[i+1]),len(sub_dic[i+2])):
                item = i
            elif len(sub_dic[i+1])>len(sub_dic[i+2]):
                item = i+1
            else:
                item = i+2            
            new_dic[k] = sub_dic[item]
            k+=1

                
    with open(dic2+name+'_min_10km.json','w') as outfile:
        json.dump(new_dic, outfile, ensure_ascii=False)

 

if __name__ == '__main__':

        
    name ='12'
    remove_seconds_improve(name)
    
    
    with open(dic2+name+'_min_10km.json','r') as f:
        diction1 = json.load(fp=f)

    
    l=[]
    for i in range(len(diction1)):
        v = str(i)
        l.append(len(diction1[v]))
    
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims)
    plt.plot(l)
    plt.xlabel('Time')
    plt.ylabel('Vehicle Number')
    plt.savefig(dic2+name+'_min_10km.eps', dpi=400)
