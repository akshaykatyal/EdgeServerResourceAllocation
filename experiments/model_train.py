#making all the necessary imports to run the trainner
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

#defining function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser("Resource allocation in multi-agent environment")
    # Environment
    parser.add_argument("--scenario", type=str, default="resourceallocate", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=2, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=2500, help="number of episodes")
    parser.add_argument("--step-num", type=int, default=8, help="number of time steps in one slot")
    parser.add_argument("--testing-interval", type=int, default=10, help="testing interval")
    parser.add_argument("--learning-type", type=str, default="maddpg", help="agent learning policy")
    parser.add_argument("--reward-maddpg", default=True, help="reward is the minimum of delay")
    parser.add_argument("--Q-type", type=str, default="finite", help="queue type for edge server")

    # Core training parameter10
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for the Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    #the number of update intervals
    parser.add_argument("--update-interval", type=int, default=10, help="update interval")
    #no of units for algorithm training
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--Group-traffic", type=int, default=7, help="Number of traffic Group")
    #the ditribution of the layers for the training of model
    parser.add_argument("--type-distribution", type=str, default="softmax", help="The distribution of action: softmax or sigmoid")
    
    # Checkpointing for the results
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="benchmark to save model after the mentioned number is complete")
    parser.add_argument("--print-rate", type=int, default=30, help="number of episodes print")
    parser.add_argument("--load-dir", type=str, default="./Restore/", help="directory in which training state and model are loaded")
    #simulation at the time interval for the delay
    epoch = 'time%.6f/' % time.time()
    folder = 'EDGESERVER_MADDPG'+epoch.replace('.', '')
    parser.add_argument("--folder", type=str, default="./ResourceResults/"+folder, help="name of the floder script")
    # Evaluation of the results
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./Results/"+folder+"figures/", help="directory where plot data is saved")
    parser.add_argument("--data-dir", type=str, default="./Results/"+folder+"data/", help="directory where data is saved")
    parser.add_argument("--vehicle-data-dir", type=str, default="./IOV_DATA/IOV_DATA_48_2/", help="directory of vehicle data")
# storing of the results fro the model trains and plots
    parser.add_argument("--MATRIX_TOPOLOGY-dir", type=str, default="./IOV_DATA/MATRIX_TOPOLOGY/", help="directory of the vehicle topology data")
    parser.add_argument("--Handover", default=False, help ="Condidered handover in assignment")
    parser.add_argument("--Que-obs", default=True, help ="Condidered agent.state.Q_delay in assignment")
    parser.add_argument("--Step-observation", default=False, help ="Condidered step in observation")
    return parser.parse_args()
#the function for multilayer perceptron
def mlp_model(input, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        #4 layers are used
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=int(num_units/2), activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=int(num_units/2), activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn= None)
        return out
    
#function to create the environment from the environment class
def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load the resource allocation scenario
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world, test_vehicles 
    world = scenario.make_world(arglist)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(arglist, world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(arglist, world, scenario.reset_world, scenario.reward, scenario.observation)
    #this will multi agent environment as output
    return env
#now get the trainers required for training the environment
def get_trainers(env, obs_shape_n, type_m):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(env.n):
        trainers.append(trainer(
            type_m+"agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(type_m == 'ddpg')))
    return trainers
#Now when the laring algorithm is maddpg
def train(arglist, learning_type = 'maddpg'):
    with U.single_threaded_session():
        # Create environment , test_vehicles
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Creating environment trainers using vehicle agents
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
       #trainer of  learning of type maddpg and ddpg
        trainers = get_trainers(env, obs_shape_n, learning_type)
        #using policy based on the type of algorithm
        print('Policy used for training{}'.format( learning_type))
        # Initialize
        U.initialize()
        # Load previous results saved in the directory
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore:
            print('Loading results.....')
            U.load_state(arglist.load_dir)
      #train steps for the agents
        train_step = 0
        #store the loss of the agent
        loss_store = []
#storeing the reward for each agent in array
        Reward_all =[]
        #this will save the tensorflow training model
        saver = tf.train.Saver()
        episode = 0
        #getting delays for the reward
        delay_mean = []
        delay_max = []
        #getting mean of the delays for the rewards for the agent
        delay_fix_mean = []
        delay_fix_max = []
        Changed_number =[]
        changed_group_num_all=[]
        #average
        delay_group_mean_ma =[]
        delay_group_mean_fix =[]
        #calulating of delays after adding loads of server
        delay_group_mean_load_ma=[]
        delay_group_mean_load_fix=[]
        delay_group_max_ma=[]
        delay_group_max_fix =[]
        #testing in group for the delay
        delay_in_group_test = []
        #delay for one vehicle in range of server
        delay_one_agent=[]
        delay_one_agent_max =[]
        delay_one_agent_mean = []
        
        Delay_group_last =[]
        Delay_in_group =[]
        l_a=5
        l_a_r=100
        Tdelay=l_a/l_a_r

        rho = []

        print('Starting training iterations for resource allocation environment.....')
        flag = False
        sum_service_rate = 10*4
        while episode < arglist.num_episodes:  
            flag = True
            for step in range(arglist.step_num):
                #getting the observation
                obs_n = env.reset(episode, step, arglist)
                if step>0:
                    for i, agent in enumerate(trainers):
                        # replay buffer adding
                        agent.experience(last_obs_n[i], action_n[i], rew_n[i], obs_n[i], done_n[i])
#
                # Getting agent actions
                action_n = [agent.action(obs, flag) for agent, obs in zip(trainers, obs_n)]
                last_obs_n, rew_n, done_n, info_n, changed_number, changed_group_num, delay_in_group, reward_real = env.step(action_n, step, learning_type)
                #number of traing steps
                train_step += 1
                #totaling the delay in environment
                Delay_group_last.append(env.world.last_delay.flatten())
                Delay_in_group.append(delay_in_group.flatten())
         #adding up all the delays in world evironment and appending them
                Reward_all.append(np.mean(reward_real)*50)
                Changed_number.append(changed_number)
                delay_mean.append(np.mean(env.world.delay_ma))
                delay_max.append(np.max(env.world.delay_ma))
                delay_group_mean_ma.append(np.mean(delay_in_group))
                delay_group_mean_load_ma.append(np.sum(delay_in_group*env.world.load_group)/np.sum(env.world.load_group))
                delay_group_max_ma.append(np.max(delay_in_group))
                changed_group_num_all.append(changed_group_num)
                
                # now getting the distance of the vehicle from edge servers
                if learning_type == 'maddpg':
                    delay_fix_mean.append(np.mean(env.world.delay_fix))
                    delay_fix_max.append(np.max(env.world.delay_fix))
                    delay_group_mean_fix.append(np.mean(env.world.delay_in_group_fix))
                    #getting the total group delay from the world class
                    delay_group_mean_load_fix.append(np.sum(env.world.delay_in_group_fix*env.world.load_group)/np.sum(env.world.load_group))
                    delay_group_max_fix.append(np.max(env.world.delay_in_group_fix ))
                    #gettig delay for each agent
                    delay_one_agent.append(np.max(env.world.centralized_Delay))
                    delay_one_agent_max.append(np.max(env.world.centralized_Delay))
                    delay_one_agent_mean.append(np.mean(env.world.centralized_Delay))
                    rho_ = np.sum(env.world.load_group)/sum_service_rate
                    rho.append(rho_)
                    
            #now if the learning is of ddpg type
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], obs_n[i], 1)   
            
            print("Delay greater > 20ms: {}".format(sum(i < -20 for i in Reward_all[-arglist.step_num:])))
            #prnting the steps episode, mean, rewardand the max reward based on delay
            print("Group:{}, steps: {}, episodes: {}, mean {} reward: {}, and max reward: {}".format(
                episode%arglist.Group_traffic, train_step, episode, learning_type, np.mean(Reward_all[-arglist.step_num:-1]), np.min(Reward_all[-arglist.step_num:-1])))
#
            if episode%arglist.update_interval ==0:
                loss = None 
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)
                    if loss != None:
                        print("Updated loss is: {}".format(loss[1]))
                        loss_store.append(loss[1])
                        print("Loss of Critic supporting actor {}".format(loss[0]))
            #appending each episode
            episode += 1
            if episode>0.3*arglist.num_episodes and episode%500==0:
                plot_implot(arglist, Reward_all, None, None, arglist.step_num, arglist.Group_traffic, "Reward received")
        
        U.save_state(arglist.data_dir, saver=saver)
    #now if the learning type is maddpg
    if learning_type == 'maddpg':
        return (Reward_all, delay_mean, delay_max, delay_fix_mean, delay_fix_max,
            Changed_number, changed_group_num_all, delay_group_mean_ma,
            delay_group_mean_fix, delay_group_mean_load_ma, delay_group_mean_load_fix,
            delay_group_max_ma, delay_group_max_fix, delay_one_agent,
            delay_in_group_test, Delay_group_last, Delay_in_group,
            Tdelay, delay_one_agent_mean,delay_one_agent_max,rho)
    else:
        return (Reward_all, delay_mean, delay_max, Changed_number, changed_group_num_all,
                delay_group_mean_ma, delay_group_mean_load_ma, delay_group_mean_load_fix,
                delay_group_max_ma, delay_in_group_test)
#Now testing the model by running the trainers
def test(trainers,env, learning_type, arglist, flag):
    delay_in_group_mean = np.zeros(9)
    for i in range(9):
        delay_in_group_max = np.zeros(arglist.step_num)
        for step in range(arglist.step_num):
            #Now testing for each episode
            episode = i+1
            obs_n = env.reset(episode, step, arglist)
            action_n = [agent.action(obs, flag) for agent, obs in zip(trainers, obs_n)]
            last_obs_n, rew_n, done_n, info_n, changed_number, changed_group_num, delay_in_group, reward_real = env.step(action_n, step, learning_type)
            delay_in_group_max[step] = np.max(delay_in_group)
        #calculating delay in resource allocation in a group
        delay_in_group_mean[i] = np.mean(delay_in_group_max)
    return delay_in_group_mean
#function to plot delay reward against time
def plot_figure(arglist, data, y_value, INTERVAL_STEP = 600):
    time = np.array(range(len(data)))
    time_e = np.trunc(time/INTERVAL_STEP)
    time_i = time_e.astype(int)
    #reading the dataframe to get the csv file
    dataframe = pd.DataFrame({'Episode': time_i, y_value:data})
    dataframe.to_csv(arglist.data_dir+ y_value + "Compare_data.csv", index=False, sep=',')
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims,edgecolor='red')
    tips = pd.read_csv(arglist.data_dir+ y_value + "Compare_data.csv")
   #plot for reward based on the number of episodes
    plot = sns.boxplot(x="Episode Number", y=y_value, data=tips, linewidth=1.5)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=18)
    plt.xlabel('Episode Numbers')
    plt.ylabel(y_value)
    plot.figure.savefig(arglist.plots_dir+y_value+ 'figure.eps', dpi=400)

def plot_implot(arglist, data, data2, data3, num1, num2, name, x_label ="Episode Numbers" ):
    # num1 is interval
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims,edgecolor='red')
    data_new = []
    if num1>1:
        for i in range(0,len(data),num1):
            data_new.append(np.mean(data[i:i+num1]))
    else:
        data_new = data
    #episode is the length of data / num2
    Episode = int(len(data_new)/num2)
    data_new = np.reshape(data_new[:num2*Episode],(Episode, num2)).T

    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18) 
    plt.rc('axes', labelsize=18) 
    plt.rc('axes', titlesize=18) 
    plt.rc('legend', fontsize=18)
    #plot of the data
    l1 = sns.tsplot(data_new, color='blue',legend = True)
    
    if data2 is not None:
        data_new2 = []
        if num1>1:
            for i in range(0,len(data2),num1):
                data_new2.append(np.mean(data2[i:i+num1]))
        else:
            data_new2=data2
         #getting the shape of new data after appending
        data_new2 = np.reshape(data_new2[:num2*Episode],( Episode, num2)).T
        l2 = sns.tsplot(data_new2, color='red',legend = True)
        if data3 is not None:
            data_new3 = []
            for i in range(0,len(data3),num1):
                data_new3.append(np.mean(data3[i:i+num1]))
        else:        
            plt.legend([l1,l2], labels=["MADDPG","DDPG"])
    else:
         plt.legend(l1, labels=["MADDPG"])
        
    plt.xlabel(x_label)
    plt.ylabel(name)
    
    l1.figure.savefig(arglist.plots_dir+ name+ 'figure.pdf', dpi=400)
#FUNCTION TO SAVE FILE TO CSV FOR THE DATA
def save_file(varaible, name, leaening_type):
    rew_file_name = arglist.data_dir + learning_type + name+'.csv'
    with open(rew_file_name, 'w') as fp:
        csv_write = csv.writer(fp)
        csv_write.writerow(varaible)
        
 
def writeList2CSV(myList, name, learning_type):
    rew_file_name = arglist.data_dir+ learning_type + name+'.csv'
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
        print("File error please check format")
    finally:
        file.close();
        
#function to plot and save as pdf
def plot_fig_one_line(data1 ,x_label, y_label, name):
    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims,edgecolor='red')
    plt.plot(data1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(arglist.plots_dir+name+ '.pdf', dpi=400)

def plot_Resource_delay(data1, data2, legend, name, max_=None):

    a4_dims = (8, 5)
    plt.figure(figsize=a4_dims,edgecolor='red')
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
    plt.xlabel('Resource Allocation Delay')
    plt.ylabel('Probability')
    plt.savefig(arglist.plots_dir+name+ '.pdf', dpi=400)
    

if __name__ == '__main__':

    arglist = parse_args()
    os.makedirs(arglist.plots_dir, exist_ok=True)
    os.makedirs(arglist.data_dir, exist_ok=True)
    t_start = time.time()

    learning_type = arglist.learning_type
    if learning_type == 'maddpg':
        (Reward_all, delay_mean, delay_max, delay_fix_mean, delay_fix_max,
            Changed_number, changed_group_num_all, delay_group_mean_ma,
            delay_group_mean_fix, delay_group_mean_load_ma, delay_group_mean_load_fix,
            delay_group_max_ma, delay_group_max_fix, delay_one_agent,
            delay_in_group_test, Delay_group_last, Delay_in_group,
            Tdelay, delay_one_agent_mean,delay_one_agent_max,rho)= train(arglist, learning_type)
    else:
        (Reward_all, delay_mean, delay_max, Changed_number, changed_group_num_all,
                delay_group_mean_ma, delay_group_mean_load_ma, delay_group_mean_load_fix,
                delay_group_max_ma, delay_in_group_test)= train(arglist, learning_type)
   # saving of the data for verification
    save_file(Reward_all,'_rewards',learning_type)
    save_file(delay_mean,'_mean_delay',learning_type)
    save_file(delay_max,'_max_delay',learning_type)
    save_file(Changed_number,'_Changed_number',learning_type)
    save_file(changed_group_num_all,'_changed_group_num_all',learning_type)
    save_file(delay_group_mean_ma,'_delay_group_mean',learning_type)
    save_file(delay_group_max_ma,'_delay_group_max_ma',learning_type)
    
    writeList2CSV(delay_in_group_test,'_delay_in_group_test',learning_type)
    save_file(delay_group_mean_load_ma,'_delay_group_mean_load_ma',learning_type)
    #saving all the delays for maddpg type
    if learning_type == 'maddpg':
        save_file(delay_fix_mean,'_mean_fix_delay','fix')
        save_file(delay_fix_max,'_max_fix_delay','fix')
        save_file(delay_group_mean_fix,'_mean_group_fix_delay','fix')
        save_file(delay_group_mean_load_fix,'_delay_group_mean_load_fix','fix')
        save_file(delay_group_max_fix,'_delay_group_max_fix','fix')
        save_file(delay_one_agent,'_delay_group_one','centrialized')
        save_file(delay_one_agent_mean,'_delay_mean','centrialized')
        save_file(delay_one_agent_max,'_delay_max','centrialized')
        
        save_file(Delay_group_last,'_Delay_group_last','maddpg')
        save_file(Delay_in_group,'_Delay_in_group','maddpg')
        save_file(rho,'_rho','sum')
        
        
          

        num = arglist.step_num
        #plot for reward
        plot_implot(arglist, Reward_all, None, None, num, arglist.Group_traffic, "Reward Received")
        #plot for delay
        plot_implot(arglist, delay_mean, delay_fix_mean, delay_one_agent_mean, num, arglist.Group_traffic, "Agent Delay Mean")
        #plot for delay in group
        plot_implot(arglist, delay_group_max_ma, delay_group_max_fix, delay_one_agent_max, num, arglist.Group_traffic, "Group Resource delay")
         
    
        data1 = [i[8] for i in delay_in_group_test]
        plot_fig_one_line(data1,'Time', 'Mean MAX Delay Testing', 'mean of the delay')
    
        matrix=np.array(delay_in_group_test,dtype=float)
        matrix_T=np.transpose(matrix)


        size = arglist.Group_traffic
        n_new =np.mean( np.reshape(Changed_number[-num*size:],(int(len(Changed_number[-num*size:])/num),num)),0)
        plot_fig_one_line(n_new ,'Time', 'VEHICLES CHANGED IN SERVER', 'Number of Vehicles')
        
        n_new =np.mean( np.reshape(changed_group_num_all[-num*size:],(int(len(changed_group_num_all[-num*size:])/num),num)),0)
        plot_fig_one_line(n_new ,'Time', 'Changed groups', 'Change Groups')
        
#Now plotting resource delay graph for the edge server
        import itertools
        data1 = list(itertools.chain(*Delay_in_group[-num*7:]))
        data2 = list(itertools.chain(*Delay_group_last[-num*7:]))
        a1 = [i for i in data1 if i>0.1]
        a2 = [i for i in data2 if i>0.1]

        plot_Resource_delay(np.array(a1), np.array(a2), [ "Re-assign","Last"], 'Resource  Assign Delay', 42)

        #plot dor time and the rho for the delay
        plot_fig_one_line(rho[-num:],'time', 'rho', 'delay -rho')
    else:
        #else plot reward against episode graph
        plot_fig_one_line(Reward_all[0:-1:10] ,'Episode Number', 'Reward received', 'reward')
    #give the time cost for the delay in resource allocation 
    print("Time cost in min: " + str(round((time.time()-t_start)/60/60, 2)))
