#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:08:33 2019

@author: tyuan
"""
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dir1 = "./Results/MADDPG/data/"
dir2 = "./Results/DDPG/data/"

def read_file(l_type, name, dird):
    
    rew_file_name = dird + l_type + name+'.csv'
    with open(rew_file_name) as fp:
        f_tsv = csv.reader(fp)
        data = next(f_tsv)  
    return data

def readCSV2List(l_type, name, dird):
    rew_file_name = dird + l_type + name+'.csv'
    try:
        file=open(rew_file_name,'r',encoding="gbk")# 读取以utf-8
        context = file.read() # 读取成str
        list_result=context.split("\n")#  以回车符\n分割成单独的行
        #每一行的各个元素是以【,】分割的，因此可以
        length=len(list_result)
        for i in range(length):
            it =[]
            for item in list_result[i].split(","):
                if item =='':
                    continue
                it.append(float(item))
            list_result[i] =it
        return list_result
    except Exception :
        print("文件读取转换失败，请检查文件路径及文件编码是否正确")
    finally:
        file.close();# 操作完成一定要关闭
        


def plot_implot(arglist, data, data2, data3, num1, num2, name, x_label ="Episode" ):
    # num1 is interval
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims)

    data_new = []
    if num1>1:
        for i in range(0,len(data),num1):
            data_new.append(np.mean(data[i:i+num1]))
    else:
        data_new = data
    Episode = int(len(data_new)/num2)
    data_new = np.reshape(data_new[:num2*Episode],(Episode, num2)).T

    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18) 
    plt.rc('axes', labelsize=18) 
    plt.rc('axes', titlesize=18) 
    plt.rc('legend', fontsize=18)
    l1 = sns.tsplot(data_new, color='red',legend = True)
    
    if data2 is not None:
        data_new2 = []
        if num1>1:
            for i in range(0,len(data2),num1):
                data_new2.append(np.mean(data2[i:i+num1]))
        else:
            data_new2=data2
        data_new2 = np.reshape(data_new2[:num2*Episode],( Episode, num2)).T
        l2 = sns.tsplot(data_new2, color='green',legend = True)
        if data3 is not None:
            data_new3 = []
            for i in range(0,len(data3),num1):
                data_new3.append(np.mean(data3[i:i+num1]))
            data_new3 = np.reshape(data_new3[:num2*Episode],( Episode, num2)).T
            l3 = sns.tsplot(data_new3)
            plt.legend([l1, l2, l3], labels=["MADDPG", "DDDPG", "DB"])
        else:        
            plt.legend([l1, l2], labels=["MADDPG", "DDDPG"])
    else:
         plt.legend(l1, labels=["MADDPG"])
        
    plt.xlabel(x_label)
    plt.ylabel(name)

    l1.figure.savefig('./Results/Compare/'+ name+ 'lineRe_figure.pdf', dpi=400)
    l1.figure.savefig('./Results/Compare/'+ name+ 'line_figure.png', dpi=400)


if __name__ == '__main__':
    #
    l_type = 'maddpg'
    rewards_maddpg = list(map(float, read_file(l_type, '_rewards',dir1)))
    max_delay_maddpg = list(map(float, read_file(l_type, '_max_delay',dir1)))
    delay_mean_maddpg = list(map(float, read_file(l_type, '_mean_delay',dir1)))
    Changed_number_maddpg = list(map(float, read_file(l_type, '_Changed_number',dir1)))
#    changed_area_num_maddpg = list(map(float, read_file(l_type, '_changed_area_num_all',dir1)))
    delay_area_mean_maddpg = list(map(float, read_file(l_type, '_delay_area_mean',dir1)))
#    delay_area_max_ma_maddpg = list(map(float, read_file(l_type, '_delay_area_max_ma',dir1)))
#    delay_in_area_test_maddpg = list(map(float, read_file(l_type, '_delay_in_area_test',dir1)))
#    _delay_area_mean_load_ma_maddpg = list(map(float, read_file(l_type, '_delay_area_mean_load_ma',dir1)))

    l_type = 'ddpg'
    rewards_ddpg = list(map(float, read_file(l_type, '_rewards',dir2)))
    max_delay_ddpg = list(map(float, read_file(l_type, '_max_delay',dir2)))
    delay_mean_ddpg = list(map(float, read_file(l_type, '_mean_delay',dir2)))
    Changed_number_ddpg = list(map(float, read_file(l_type, '_Changed_number',dir2)))
    changed_area_num_ddpg = list(map(float, read_file(l_type, '_changed_area_num_all',dir2)))
    delay_area_mean_ddpg = list(map(float, read_file(l_type, '_delay_area_mean',dir2)))
    delay_area_max_ma_ddpg = list(map(float, read_file(l_type, '_delay_area_max_ma',dir2)))
   
#    delay_in_area_test_ddpg = []
#    data = read_file(l_type, '_delay_in_area_test',dir2)
#    for i in range(len(data)):
#        delay_in_area_test_ddpg.append(list( map(float,  data[i][1:-1])))
    delay_in_area_test_ddpg = readCSV2List(l_type, '_delay_in_area_test',dir2)
#    delay_in_area_test_ddpg = list(map(float, read_file(l_type, '_delay_in_area_test',dir2)))
    delay_area_mean_load_ma_ddpg = list(map(float, read_file(l_type, '_delay_area_mean_load_ma',dir2)))
    
    l_type = 'fix'
    delay_fix_mean= list(map(float, read_file(l_type, '_mean_fix_delay',dir1)))
    delay_fix_max= list(map(float, read_file(l_type, '_max_fix_delay',dir1)))
    delay_area_mean_fix = list(map(float, read_file(l_type, '_mean_area_fix_delay',dir1)))
#    delay_area_mean_load_fix = list(map(float, read_file(l_type, '_delay_area_mean_load_fix',dir1)))
#    delay_area_max_fix = list(map(float, read_file(l_type, '_delay_area_max_fix',dir1)))
 
    
    cut =-400
    plot_implot(arglist, rewards_maddpg[:cut], rewards_ddpg[:cut], None, arglist.step_num, arglist.Group_traffic, "Reward")
    plot_implot(arglist, max_delay_maddpg[:cut], max_delay_ddpg[:cut], delay_fix_max[:cut], arglist.step_num, arglist.Group_traffic, "Max area Delay")
    
    
    
    
    
    