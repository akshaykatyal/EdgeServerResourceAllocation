#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:46:07 2019

@author: tyuan
to get data of brezil about vehicles location
"""
import numpy as np

import json

uni_time = list(np.load('/Users/eva/Documents/GitHub/vehicle_data_bresil/time_30.npy'))

for i in range(1,11):
    if i <10:
        name = '0'+str(i)
    else:
        name = str(i)      
    fname_sub = '2014-10-'+name+'.txt'
    dic = '/Users/eva/Documents/GitHub/vehicle_data_bresil/'
#    fname_sub = 'test.txt'
    fname = dic +fname_sub
    #with open(, 'r') as f:

#    s=[]
    d ={}
    x = []
    with open(fname, 'r', encoding='ascii') as f:
        for je in f.readlines():
            tm = je[:-1].split(',')
            if tm[1] in uni_time:
#                s.append has_key(tm)
                if tm[1] in d.keys():
                    d[tm[1]].append(tm[4:6])
                else:
                    d[tm[1]] = [tm[4:6]]
                    
                x.append(tm[4:6])
    x_tem = np.array(x)
    print(min(x_tem[:,0]))
    
    print(max(x_tem[:,0]))
    
    print(min(x_tem[:,1]))
    print(max(x_tem[:,1]))
    
    

#        s = [j[:-1].split(',') for j in f.readlines()]

            
#    list_ndarray = np.array(s)
#    d={}
##    if i ==1:
##        uni_time = np.unique(list_ndarray[:,1])[0:-1:600]
#
#    for time in uni_time:
#        d[time] = list_ndarray[list_ndarray[:,1] == time][:,4:6].tolist()
#    
#    np.save(dic+'dic'+name+'.npy',d)   
    with open(dic+'dic_30'+name+'.json','w') as outfile:
        json.dump(d, outfile, ensure_ascii=False)
#        outfile.write('\n')
        
#        
#    with open(dic+'dic'+name+'.json', 'r') as f:
#        diction = json.load(fp=f)


