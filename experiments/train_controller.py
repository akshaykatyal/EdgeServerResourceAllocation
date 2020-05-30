
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:36:22 2019

@author: tyuan
"""
import sys 
sys.path.append("..") 
import argparse
import numpy as np
import tensorflow as tf
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import csv
#from scipy.stats import cumfreqs
#from maddpg.plot_file.plot_3d import plot_3d_figure

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_controller", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=2, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=4000, help="number of episodes")
    parser.add_argument("--step-num", type=int, default=48, help="number of time step in one day")
    parser.add_argument("--testing-interval", type=int, default=10, help="testing interval")
#    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
#    parser.add_argument("--adv-policy", type=str, default="maddpg", help="poxlicy of adversaries")
    parser.add_argument("--l-type", type=str, default="maddpg", help="poxlicy of learning")
    parser.add_argument("--reward-same", default=True, help="ddpg reward is maxdelay of all areas or not")
    parser.add_argument("--Q-type", type=str, default="finite", help="Queue type, inf and finite") #"inf"

    # Core training parameter10
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
#    parser.add_argument("--lr-discount", type=float, default=0.995, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--update-interval", type=int, default=7, help="update interval")    
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--Group-traffic", type=int, default=7, help="Number of traffic Group")
    parser.add_argument("--type-distribution", type=str, default="softmax", help="The distribution of action: softmax or sigmoid")
    
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--print-rate", type=int, default=30, help="prin episodes")
    parser.add_argument("--load-dir", type=str, default="./Restore/", help="directory in which training state and model are loaded")
    epoch = 't%.6f/' % time.time()
    folder = 'DDPG_48_C02_'+epoch.replace('.', '')
    parser.add_argument("--folder", type=str, default="./Results/"+folder, help="name of the floder script")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./Results/"+folder+"figures/", help="directory where plot data is saved")
    parser.add_argument("--data-dir", type=str, default="./Results/"+folder+"data/", help="directory where data is saved")
    parser.add_argument("--vehicle-data-dir", type=str, default="./DATA/SDNdata_48_2/", help="directory of vehicle data")
#                        SDNdata_24_2/
    parser.add_argument("--topo-dir", type=str, default="./DATA/topo/", help="directory of topology data")
    parser.add_argument("--Handover", default=False, help ="Condidered handover in assignment")
    parser.add_argument("--Que-obs", default=True, help ="Condidered agent.state.Q_delay in assignment")
    parser.add_argument("--Step-ob", default=False, help ="Condidered step in observation")
    parser.add_argument("--Plr-obs", default=False, help ="Condidered packet loss rate in observation")


    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=int(num_units/2), activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=int(num_units/2), activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn= None)
#                                     tf.nn.sigmoid) 
        return out
    

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.multiagent.environment import MultiAgentEnv
    import multiagent.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world, test_vehicles 
    world = scenario.make_world(arglist)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(arglist, world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(arglist, world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
#, test_vehicles

def get_trainers(env, obs_shape_n, type_m):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(env.n):
        trainers.append(trainer(
            type_m+"agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(type_m == 'ddpg')))
    return trainers


def train(arglist, l_type = 'maddpg'):
    with U.single_threaded_session():
        # Create environment , test_vehicles
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        #trainer of maddpg and ddpg
        trainers = get_trainers(env, obs_shape_n, l_type)
#        trainers_ddpg = get_trainers(env, obs_shape_n, 'ddpg')
        
        print('Using policy {}'.format( l_type))
        # Initialize
        U.initialize()
        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

#        episode_rewards = [0.0]  # sum of rewards for all agents
        train_step = 0

        loss_store = []

        Reward_all =[]
        saver = tf.train.Saver()
        episode = 0
        delay_mean = []
        delay_max = []

        delay_fix_mean = []
        delay_fix_max = []
        Changed_number =[ ]   
        changed_area_num_all=[]
        
        # 全部区域的平均延时
        delay_area_mean_ma =[]
        delay_area_mean_fix =[]

        #加入load后，全部区域的平均延时
        delay_area_mean_load_ma=[]
        delay_area_mean_load_fix=[]
        delay_area_max_ma=[]
        delay_area_max_fix =[]
        
        #测试组
        delay_in_area_test = []
        
        #只有一个agent
        delay_one_agent=[]
        delay_one_agent_max =[]
        delay_one_agent_mean = []
        
        Delay_areas_last =[]
        Delay_in_area =[]
        PLR = []
        
        Plr_fix = []
        Plr_one_agent = []
        rho = []
        
        
#        if arglist.restore:
#             radom_flag = False
#        else:
        
        print('Starting iterations...')
        #First Testing
        radom_flag = False
        sum_serve_rate = 10*4
        #every 80 times for one testing
#        delay_in_area_test.append(test(trainers,env, l_type, arglist, radom_flag))
        while episode < arglist.num_episodes:  
            radom_flag = True
#            if episode < int(0.8*arglist.num_episodes):
#                radom_flag = True  
#            else:
#                radom_flag = False
            #step
            
            for step in range(arglist.step_num):
                obs_n = env.reset(episode, step, arglist)
                if step>0:
                    for i, agent in enumerate(trainers):
                        # add buffer
                        agent.experience(last_obs_n[i], action_n[i], rew_n[i], obs_n[i], done_n[i])
#                        sum_serve_rate
                # get action
                action_n = [agent.action(obs, radom_flag) for agent, obs in zip(trainers, obs_n)]
                last_obs_n, rew_n, done_n, info_n, changed_number, changed_area_num, delay_in_area, reward_real, plr = env.step(action_n, step, l_type)
                
                train_step += 1
                             
                Delay_areas_last.append(env.world.last_delay.flatten())
                Delay_in_area.append(delay_in_area.flatten())
                
#                print(np.mean(reward_real)*20)
                Reward_all.append(np.mean(reward_real)*30)
                Changed_number.append(changed_number)
                delay_mean.append(np.mean(env.world.delay_ma))
                delay_max.append(np.max(env.world.delay_ma))
                delay_area_mean_ma.append(np.mean(delay_in_area))
                delay_area_mean_load_ma.append(np.sum(delay_in_area*env.world.load_areas)/np.sum(env.world.load_areas))
                delay_area_max_ma.append(np.max(delay_in_area))
                PLR.append(np.mean(plr))
                

                
                
                changed_area_num_all.append(changed_area_num)
                
                # data for distance based method 
                if l_type == 'maddpg':
                    delay_fix_mean.append(np.mean(env.world.delay_fix))
                    delay_fix_max.append(np.max(env.world.delay_fix))
                    delay_area_mean_fix.append(np.mean(env.world.delay_in_area_fix))
                    delay_area_mean_load_fix.append(np.sum(env.world.delay_in_area_fix*env.world.load_areas)/np.sum(env.world.load_areas))
                    delay_area_max_fix.append(np.max(env.world.delay_in_area_fix ))
                    delay_one_agent.append(np.max(env.world.centralized_Delay))
                    delay_one_agent_max.append(np.max(env.world.centralized_Delay))
                    delay_one_agent_mean.append(np.mean(env.world.centralized_Delay))

                    Plr_fix.append(np.mean(env.world.Plr_fix))
                    Plr_one_agent.append(env.world.Plr_centralized)
                    
                    rho_ = np.sum(env.world.load_areas)/sum_serve_rate
                    rho.append(rho_)
                    
#            if l_type == 'ddpg':
            # add experience of last step
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], obs_n[i], 1)   
            
            print("delay more than 20 ms: {}".format(sum(i < -20 for i in Reward_all[-arglist.step_num:])))
            
            print("G:{}, steps: {}, episodes: {}, mean {} reward: {}, and max reward: {}".format(
                episode%arglist.Group_traffic, train_step, episode, l_type, np.mean(Reward_all[-arglist.step_num:-1]), np.min(Reward_all[-arglist.step_num:-1])))
#            else:
#                print("steps: {}, episodes: {}, maddpg reward: {}, ddpg reward: {}".format(
#                    train_step, episode, np.mean(Reward_all[train_step-arglist.step_num:train_step]),np.mean(Reward_ddpg[-arglist.step_num:-1])))
            if episode%arglist.update_interval ==0:
                loss = None 
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)
                    if loss != None:
                        print("Updating and the loss is: {}".format(loss[1]))
                        loss_store.append(loss[1])
                        print("Updating and the loss of critic is: {}".format(loss[0]))
    #                    if radom_flag ==True:
    #                       radom_flag = False
#                    if episode%arglist.testing_interval ==0:
#                        radom_flag = False
#                        #every 80 times for one testing
#                        delay_in_area_test.append(test(trainers,env, l_type, arglist, radom_flag))
            episode += 1
            if episode>0.3*arglist.num_episodes and episode%500==0:
                plot_implot(arglist, Reward_all, None, None, arglist.step_num, arglist.Group_traffic, "Reward")
#                plot_implot(arglist, delay_mean, delay_fix_mean, delay_one_agent_mean, arglist.num_episodes, arglist.Group_traffic, "Mean Agent Delay")
            
#            if l_type == 'maddpg' and (episode % arglist.save_rate == 0):
#                    U.save_state(arglist.data_dir+str(episode)+"/", saver=saver)
        
        U.save_state(arglist.data_dir, saver=saver) 
#           
    if l_type == 'maddpg':
        return (Reward_all, delay_mean, delay_max, delay_fix_mean, delay_fix_max,
            Changed_number, changed_area_num_all, delay_area_mean_ma, 
            delay_area_mean_fix, delay_area_mean_load_ma, delay_area_mean_load_fix,
            delay_area_max_ma, delay_area_max_fix, delay_one_agent,
            delay_in_area_test, Delay_areas_last, Delay_in_area, 
            PLR, Plr_fix, Plr_one_agent,
            delay_one_agent_mean,delay_one_agent_max,
            rho)
    else:
        return (Reward_all, delay_mean, delay_max, Changed_number, changed_area_num_all,
                delay_area_mean_ma, delay_area_mean_load_ma, delay_area_mean_load_fix,
                delay_area_max_ma, delay_in_area_test, PLR)

    
    

def test(trainers,env, l_type, arglist, radom_flag):
    delay_in_area_mean = np.zeros(9)
    for i in range(9):
        delay_in_area_max = np.zeros(arglist.step_num)
        for step in range(arglist.step_num):
            # get action
    #        episode = arglist.Group_traffic+1
            episode = i+1
            obs_n = env.reset(episode, step, arglist)
            action_n = [agent.action(obs, radom_flag) for agent, obs in zip(trainers, obs_n)]   
            last_obs_n, rew_n, done_n, info_n, changed_number, changed_area_num, delay_in_area, reward_real = env.step(action_n, step, l_type)
            delay_in_area_max[step] = np.max(delay_in_area)
        delay_in_area_mean[i] = np.mean(delay_in_area_max)
    return delay_in_area_mean

def plot_figure(arglist, data, y_value, INTERVAL_STEP = 600):
 
    time = np.array(range(len(data)))
    time_e = np.trunc(time/INTERVAL_STEP)
    
    time_i = time_e.astype(int)
    
    dataframe = pd.DataFrame({'Episode': time_i, y_value:data})
    dataframe.to_csv(arglist.data_dir+ y_value + "Compare_data.csv", index=False, sep=',')
    
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims)
    tips = pd.read_csv(arglist.data_dir+ y_value + "Compare_data.csv")
    
   
    plot = sns.boxplot(x="Episode", y=y_value, data=tips, linewidth=1.5)
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18) 
    plt.rc('axes', labelsize=18) 
    plt.rc('axes', titlesize=18) 
    plt.rc('legend', fontsize=18)
    plt.xlabel('Episode')
    plt.ylabel(y_value)
    plot.figure.savefig(arglist.plots_dir+y_value+ 'box_figure.eps', dpi=400)
    

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
            plt.legend([l1, l2, l3], labels=["MADDPG", "DB", "CP"])
        else:        
            plt.legend([l1, l2], labels=["MADDPG", "DB"])
    else:
         plt.legend(l1, labels=["MADDPG"])
        
    plt.xlabel(x_label)
    plt.ylabel(name)
    
    l1.figure.savefig(arglist.plots_dir+ name+ 'line_figure.pdf', dpi=400)
#    l1.figure.savefig(arglist.plots_dir+ name+ 'line_figure.png', dpi=400)

def save_file(varaible, name, l_type):
    
    rew_file_name = arglist.data_dir + l_type + name+'.csv'
    with open(rew_file_name, 'w') as fp:
        csv_write = csv.writer(fp)
        csv_write.writerow(varaible)
        
 
def writeList2CSV(myList, name, l_type):
    rew_file_name = arglist.data_dir+ l_type + name+'.csv'
    try:
        file=open(rew_file_name,'w')
        lenth_ = len(myList)
        j=0
        for items in myList:
            length = len(items)
            for i, item in enumerate(items):
                file.write(str(item))
                if i< length-1:
                    file.write(",")
            if j < lenth_- 1:
                j += 1
                file.write("\n") 
    except Exception :
        print("数据写入失败，请检查文件路径及文件编码是否正确")
    finally:
        file.close();# 操作完成一定要关闭       
        

def plot_fig_one_line(data1 ,x_label, y_label, name):
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims)
    plt.plot(data1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(arglist.plots_dir+name+ '.pdf', dpi=400)
#    plt.savefig(arglist.plots_dir+name+ '.png', dpi=400)
    
def plot_fig_two_line(data1, data2 ,x_label, y_label, legend, name):
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims)
    plt.plot(data1)
    plt.plot(data2)
    plt.legend(labels=legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(arglist.plots_dir+name+ '.pdf', dpi=400)
#    plt.savefig(arglist.plots_dir+name+ '.png', dpi=400)
    
def plot_fig_three_line(data1, data2 ,data3,x_label, y_label, legend, name):
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims)
    plt.plot(data1)
    plt.plot(data2)
    plt.plot(data3)
    plt.legend(labels=legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(arglist.plots_dir+name+ '.pdf', dpi=400)
#    plt.savefig(arglist.plots_dir+name+ '.png', dpi=400)
    
    
def plot_fig_tsplot(data1, x_label, y_label, name):    
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims)
    sns.tsplot(data1, color='red',legend = True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(arglist.plots_dir+name+ '.pdf', dpi=400)
#    plt.savefig(arglist.plots_dir+name+ '.png', dpi=400)
 
    
def plot_CDF(data1, data2, legend, name, max_=None):
#    from scipy.integrate import cumtrapz
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims)
    if max_ ==None:
        max_ =  max(np.max(data1),np.max(data2))+1
    if max_/50 < 1:
        dx = round(max_/50, 2) 
    else:
        dx = int(max_/50)
    X = np.arange(0, max_, dx)
#    data1 = np.sort(data1)
    
    y_1 = []
    y_2 = []
    for x in X:
       y_1.append(len(data1[data1<x])/len(data1))
       y_2.append(len(data2[data2<x])/len(data2))
    
    plt.plot(X, y_1, 'r')
    plt.plot(X, y_2, 'b')
    
    plt.legend(labels=legend)
    plt.xlabel('Delay')
    plt.ylabel('Cumulative Probability')
    plt.savefig(arglist.plots_dir+name+ '.pdf', dpi=400)
#    plt.savefig(arglist.plots_dir+name+ '.png', dpi=400)     
      
    

if __name__ == '__main__':

    arglist = parse_args()
    os.makedirs(arglist.plots_dir, exist_ok=True)
    os.makedirs(arglist.data_dir, exist_ok=True)
    t_start = time.time()

    l_type = arglist.l_type
    if l_type == 'maddpg':
        (Reward_all, delay_mean, delay_max, delay_fix_mean, delay_fix_max,
         Changed_number, changed_area_num_all, delay_area_mean_ma,
         delay_area_mean_fix, delay_area_mean_load_ma, delay_area_mean_load_fix,
         delay_area_max_ma, delay_area_max_fix, delay_one_agent,
         delay_in_area_test, Delay_areas_last, Delay_in_area, 
         PLR, Plr_fix, Plr_one_agent,
         delay_one_agent_mean, delay_one_agent_max, rho)= train(arglist, l_type)
    else: 
        (Reward_all, delay_mean, delay_max, Changed_number, changed_area_num_all,
            delay_area_mean_ma, delay_area_mean_load_ma, delay_area_mean_load_fix,
            delay_area_max_ma, delay_in_area_test, PLR)= train(arglist, l_type)
#                                                                          
    save_file(Reward_all,'_rewards',l_type)
    save_file(delay_mean,'_mean_delay',l_type)
    save_file(delay_max,'_max_delay',l_type)
    save_file(Changed_number,'_Changed_number',l_type)
    save_file(changed_area_num_all,'_changed_area_num_all',l_type)
    save_file(delay_area_mean_ma,'_delay_area_mean',l_type)
    save_file(delay_area_max_ma,'_delay_area_max_ma',l_type)
    
    writeList2CSV(delay_in_area_test,'_delay_in_area_test',l_type)
    save_file(delay_area_mean_load_ma,'_delay_area_mean_load_ma',l_type)
    save_file(PLR,'_PLR',l_type)
    
    if l_type == 'maddpg':
        save_file(delay_fix_mean,'_mean_fix_delay','fix')
        save_file(delay_fix_max,'_max_fix_delay','fix')
        save_file(delay_area_mean_fix,'_mean_area_fix_delay','fix')
        save_file(delay_area_mean_load_fix,'_delay_area_mean_load_fix','fix')
        save_file(delay_area_max_fix,'_delay_area_max_fix','fix')
        save_file(delay_one_agent,'_delay_area_one','centrialized')
        save_file(delay_one_agent_mean,'_delay_mean','centrialized')
        save_file(delay_one_agent_max,'_delay_max','centrialized')
        
        save_file(Delay_areas_last,'_Delay_areas_last','maddpg')
        save_file(Delay_in_area,'_Delay_in_area','maddpg')
        save_file(Plr_fix,'_PLR','fix')
        save_file(Plr_one_agent,'_PLR','Centralized')
        save_file(rho,'_rho','sum')
        
        
        
          

        num = arglist.step_num
        plot_implot(arglist, Reward_all, None, None, num, arglist.Group_traffic, "Reward")
        plot_implot(arglist, delay_mean, delay_fix_mean, delay_one_agent_mean, num, arglist.Group_traffic, "Mean Agent Delay")
        plot_implot(arglist, delay_area_max_ma, delay_area_max_fix, delay_one_agent_max, num, arglist.Group_traffic, "Max area Delay")
         
    
        data1 = [i[8] for i in delay_in_area_test] 
        plot_fig_one_line(data1,'Time', 'Mean MAX Delay Testing', 'testing')
    
        matrix=np.array(delay_in_area_test,dtype=float)
        matrix_T=np.transpose(matrix)
#        plot_fig_tsplot(matrix_T, 'Time', 'Delay Testing', 'test_delay')
    
        plot_fig_three_line(delay_area_mean_ma[-num:], delay_area_mean_fix[-num:], delay_one_agent[-num:],
                          'Time', 'Mean Delay', ["MADDPG", "BD","CP"], 'mean_delay_time3')
        
#        plot_fig_two_line(delay_area_mean_ma[-num:], delay_area_mean_fix[-num:],
#                          'Time', 'Mean Delay', ["MADDPG", "BD"], 'mean_delay_time')
#            
        
        plot_fig_three_line(delay_area_max_ma[-num:], delay_area_max_fix[-num:],delay_one_agent[-num:],
                          'Time', 'Max Delay', ["MADDPG", "BD","CP"], 'max_delay_time')
        
        size = arglist.Group_traffic
        n_new =np.mean( np.reshape(Changed_number[-num*size:],(int(len(Changed_number[-num*size:])/num),num)),0)
        plot_fig_one_line(n_new ,'Time', 'Percentage of changed vehicles', 'number_v')
        
        n_new =np.mean( np.reshape(changed_area_num_all[-num*size:],(int(len(changed_area_num_all[-num*size:])/num),num)),0)
        plot_fig_one_line(n_new ,'Time', 'Changed areas', 'number_area')
        
#        plot CDF
        import itertools
        data1 = list(itertools.chain(*Delay_in_area[-num*7:]))
        data2 = list(itertools.chain(*Delay_areas_last[-num*7:]))
        a1 = [i for i in data1 if i>0.1]
        a2 = [i for i in data2 if i>0.1]

        plot_CDF(np.array(a1), np.array(a2), [ "Re-assign","Last"], 'CDF', 42)
          
        plot_fig_three_line(PLR[-num:], Plr_fix[-num:],Plr_one_agent[-num:],
                          'Time', 'PLR', ["MADDPG", "BD","CP"], 'plr')
        
        
        plot_fig_one_line(rho[-num:],'time', 'rho', 'rho')
    else:
        plot_fig_one_line(Reward_all[0:-1:10] ,'Episode', 'Reward', 'reward')
    
    print("Time cost in min: " + str(round((time.time()-t_start)/60/60, 2)))
