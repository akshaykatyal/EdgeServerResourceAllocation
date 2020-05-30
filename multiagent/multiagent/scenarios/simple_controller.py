import numpy as np
from multiagent.multiagent.core import World, Agent, Vehicle
from multiagent.multiagent.scenario import BaseScenario
#from multiagent.opt_method import opt_assignment
import json
import networkx as nx
import math
#dic2 = '/Users/eva/Documents/GitHub/vehicle_data_bresil/'
#dic1= '/Users/tyuan/Documents/GitHub/DATA/SDNdata/'
#vehicle_data_dir

l_a = 10 # packets/ms, lower-bound of arriving rate of vehicles of G1  
l_b = 30# packets/ms  for upper-bound G1
l_a_r = 100 # packets/ms of RSUs
l_b_r = 200# packets of Q size
remote_dealy_max = 8 #ms
length_Q = 200
# D_resend = 20




class Scenario(BaseScenario):
    
    def make_world(self, arglist, groupnum = None):
        world = World()
#        # set any world properties first
#        world.dim_c = 2
#        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(world.agent_num )]
#        position = np.random.randint(0, high=world.region_W, size=[world.agent_num,2])
        
        position = (np.array([(38, 29),(40, 70),(56, 34),(72.4, 73.3)])/(100/world.region_W))
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            # position of agents
            agent.pos = tuple(position[i])

#        world.set_Radius()
        world.set_server_rate()
        world.get_crossover()
        world.set_action_dim()
        
        world.set_topo(arglist.topo_dir)
        
#        self.get_control_path_and_delay(world)
        # get path and delay in path of small areas world.area_delay
        self.get_delay_areas(world)
        # get delay of corss areas of agents
        self.get_cross_area_prog_delay(world)
        
        # make initial conditions
        self.reset_world(world, -1, 0, arglist)
        for agent in world.agents:
            agent.state.v_manage = agent.distance_manage
#            np.zeros(len(agent.cross_areas))
        
       
        
        self.Handover = arglist.Handover
        self.Que_obs = arglist.Que_obs
        self.Plr_obs = arglist.Plr_obs
        self.Group_traffic = arglist.Group_traffic
        
        self.step_num = arglist.step_num
        
        self.Step_ob = arglist.Step_ob
        
        self.reward_maddpg_same = arglist.reward_same
        
            
#    # set the test batch
#    world.test_num = int(1440/30)-1
#    test_vehicles = []
#    if groupnum == None:
#        test_vehicles = self.test_vehicle_data(world, world.test_num)
#    else:
#        for i in range(groupnum):
#            test_vehicles.append(self.test_vehicle_data(world, world.test_num))
        return world


    def reset_world(self, world, train_step, step, arglist, IF_test=False, TEST_V=None):
        # load of each block in region_H * region_W
        if train_step > - 1:
            # generte vehicle location and load with dataset
            self.get_location_vehicle(world, train_step, step, arglist.vehicle_data_dir)
        else:
#            if IF_test:
#                self.generate_vehicle_test(world, TEST_V)
#            else:
                #random generte vehicle location and load
            self.generate_vehicle(world)
        self.Q_type = arglist.Q_type
        # get load of each small areas *self.load_matrix
#        self.vehicle_to_areaload(world)
        # optimization of assignment -> delay min
        self.serve_rate =[]
        for agent in world.agents:
            self.serve_rate.append(agent.serve_rate)            
#        self.delay_p = self.get_propagate_delay_all(world)
        load = self.load.reshape(world.num_v,1)

#        print(world.load_areas)
#        print(np.sum(world.load_areas))
        for agent in world.agents:
            agent_load = []
            all_area_agent = agent.areas
            for i in range(len(all_area_agent)):
                 load = world.load_areas[all_area_agent[i]]
                 agent_load.append(load)
            #get load of agent in all its coverage areas
            agent.state.load = agent_load
            
            # get loaf of cross areas
            list_value = world.mul_area_c.get(agent.id)[0]
            agent_value_num = len(list_value)
            agent_load =[]
            for i in range(agent_value_num):
                load = world.load_areas[list_value[i]]
                agent_load.append(load)            
            agent.state.c_load = agent_load
            
            diff_agent_load = []
            # get load of fix areas
            diff_value = sorted(list(set(agent.areas).difference(set(list_value))))
            for i in range(len(diff_value)):
                load = world.load_areas[diff_value[i]]
                diff_agent_load.append(load)

            agent.state.fix_load = sum(diff_agent_load)
         # get the latency map and agent_latency of distance based method,   
        self.get_distance_based_delay(world) 
        
        # get delay of each areas based on centralizd agent.
        
        world.centralized_Delay = self.get_delay_centralzed(world)
        


#    def benchmark_data(self, agent, world):
#        rew = 0
#        collisions = 0
#        occupied_landmarks = 0
#        min_dists = 0
#        for l in world.landmarks:
#            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
#            min_dists += min(dists)
#            rew -= min(dists)
#            if min(dists) < 0.1:
#                occupied_landmarks += 1
#        if agent.collide:
#            for a in world.agents:
#                if self.is_collision(a, agent):
#                    rew -= 1
#                    collisions += 1
#        return (rew, collisions, min_dists, occupied_landmarks)


#    def is_collision(self, agent1, agent2):
#        delta_pos = agent1.state.p_pos - agent2.state.p_pos
#        dist = np.sqrt(np.sum(np.square(delta_pos)))
#        dist_min = agent1.size + agent2.size
#        return True if dist < dist_min else False
    
    def get_distance_based_delay(self, world):
        Q_delay_fix, Plr= self.get_latency_agent_fix(world, np.array(self.serve_rate))
        delay_in_area_fix = self.get_latency_area(world, Q_delay_fix, world.distance_assign_matrix)
        delay_agent_fix = np.zeros(world.agent_num)
        for i, agent in  enumerate(world.agents):  
            load_agent = world.agent_area_cover[i,:,:]*world.load_areas
            if np.sum(load_agent) <=0:
                delay_agent_fix[i] =0
            else:
                delay_agent_fix[i]=np.sum(load_agent*delay_in_area_fix)/np.sum(load_agent)

        world.delay_fix = delay_agent_fix
        
        world.delay_in_area_fix = delay_in_area_fix
        world.Plr_fix = Plr
            
    def get_delay_centralzed(self,world):
        load_c =np.sum(world.load_areas)
        world.Plr_centralized = self.PLR(load_c, np.sum(np.array(self.serve_rate)), world.agent_num*length_Q, fix_load_controller=0)
        if self.Q_type == "inf":
            Q_delay_one_agent = self.delay_Queue_inf(load_c, np.sum(np.array(self.serve_rate)), fix_load_controller=0)
        else:
            Q_delay_one_agent = self.delay_Queue(load_c, np.sum(np.array(self.serve_rate)), world.agent_num*length_Q, world.Plr_centralized, fix_load_controller=0)
        n = len(world.vehicles)/1200
        delay_area_centralzed_agent = max(2,min(remote_dealy_max, n*remote_dealy_max)) + Q_delay_one_agent + 1.3
        world.Plr_centralized = self.PLR(load_c, np.sum(np.array(self.serve_rate)), world.agent_num*length_Q, fix_load_controller=0)
        
        return delay_area_centralzed_agent
#        self.get_latency_area(world, Q_delay_one_agent, world.distance_assign_matrix)

        
    
    def reward(self, world, ltype = 'maddpg'):

        changed_num_vehicle = np.zeros(world.agent_num)
        changed_area_num = np.zeros(world.agent_num)
        Q_delay=np.zeros(world.agent_num)
        plr = np.zeros(world.agent_num)
        hand_over = np.zeros(world.agent_num)
#        print("assignment")
#        print(world.all_con_areas)
#        print("vehcile number")
#        print(world.vehicle_num)
        for i, agent in  enumerate(world.agents):         
            Q_delay[i],plr[i] = self.get_latency_one_agent(agent, world, np.array(self.serve_rate[i]))
            
            a = world.all_con_areas.copy()

            b = world.last_all_con_areas.copy()
            a[a != agent.id+1]=0
            b[b != agent.id+1]=0
            changed_areas = a - b
            changed_areas[ changed_areas<0 ]=0
            changed_areas[ changed_areas!=0]=1
            all_num = a*world.vehicle_num
            changed_area_num[i] = sum(sum(changed_areas))
        
            changed_num_vehicle[i] = sum(sum(changed_areas*world.vehicle_num))
            per_chan = max(0,min(1,changed_num_vehicle[i]/(sum(sum(all_num))+0.01)))
            x = np.argwhere(changed_areas ==1)
        
            # considered th changed number  handover cost
            if np.size(x)==0:
                hand_over[i]=0
            else:
                hand_over[i] = changed_num_vehicle[i]*0.02
#                per_chan
#                min(0.5, 0.002*changed_num_vehicle)
#                *np.mean((world.area_delay[agent.id,x[:,0],x[:,1]]))
         
        delay_in_area = self.get_latency_area(world, Q_delay, world.all_con_areas)      
        Changed_percentage_all = np.sum(changed_num_vehicle)/sum(sum(world.vehicle_num))
        delay_agent = np.zeros(world.agent_num)
#        delay_agent_fix = np.zeros(world.agent_num)
        for i, agent in  enumerate(world.agents):  
            load_agent = world.agent_area_cover[i,:,:]*world.load_areas
            if np.sum(load_agent) <=0:
                delay_agent[i] =0
            elif self.Handover:
                delay_agent[i]=np.max(world.agent_area_cover[i,:,:]*delay_in_area)
#                np.sum(load_agent*delay_in_area)/np.sum(load_agent)
#                +hand_over[i]
            else:
                delay_agent[i]=np.max(world.agent_area_cover[i,:,:]*delay_in_area)
#                np.sum(load_agent*delay_in_area)/np.sum(load_agent)

#        reward =  - delay_agent/10
        
        if self.Handover:
            reward_real =  (- np.max(delay_in_area) - np.sum(changed_area_num)/world.cross_areas_num)/30
#            Changed_percentage_all*0.5*(np.max(delay_in_area)/10)
        else:
            reward_real =  - np.max(delay_in_area)/30
#            
        if ltype == 'maddpg':
            reward = - delay_agent/30
        elif self.reward_maddpg_same: 
            reward = reward_real
        else:   
            reward =  - delay_agent/30
        
#        reward = - delay_agent/45
#        reward = np.max(world.last_delay) - np.max(delay_in_area) - hand_over
        

        world.delay_ma = delay_agent
        return (reward, Changed_percentage_all,
                np.sum(changed_area_num)/world.cross_areas_num, delay_in_area, reward_real, plr)

    def get_latency_area(self, world, Q_delay, agent):
        delay_in_area = np.zeros([world.region_W, world.region_H])
        for x in range(world.region_W):
            for y in range(world.region_H): 
                if agent[x,y]>0:
                    i = int(agent[x,y])-1
                    delay_in_area[x,y] = Q_delay[i] + world.area_delay[i,x,y]
        return delay_in_area
    
#    def get_propagate_delay_all(self, world):
#        #delay of vehicles
#        latency_p=np.zeros([world.agent_num, world.num_v])
#        for j, vehicle in enumerate(world.vehicles): 
#             dists = [np.sqrt(np.sum(np.square(world.router_pos[node] - vehicle.pos))) for node in list(world.topo_matrix.node)]
#             min_dists = dists.index( min(dists))
#             a1 = np.array(world.path_delay_p)
#             latency_p[:, j] = a1[:, min_dists]
#        return latency_p
          
        
    def get_latency_one_agent(self, agent, world, serv_rate):
        # average latency of agent based on Small areas
        
        agent_load_vector = agent.state.c_load * agent.state.v_manage
        load_all_cross = np.sum(agent_load_vector)
        plr = self.PLR(load_all_cross, serv_rate, length_Q, agent.state.fix_load)
        if self.Q_type == "inf":
            Q_delay = self.delay_Queue_inf(load_all_cross, serv_rate, agent.state.fix_load) 
            plr = 0
        else:
           
            Q_delay = self.delay_Queue(load_all_cross, serv_rate, length_Q, plr, agent.state.fix_load)
        return Q_delay, plr 
       

    def get_latency_agent_fix(self, world, serv_rate):
        # average latency of agent based on Small areas
        Q_delay = np.zeros(world.agent_num)
        Plr = np.zeros(world.agent_num)
#        packet loss rate
#        for i, agent in  enumerate(world.agents):
#            Plr[i] = self.PLR(load_all_cross, serv_rate[i], length_Q, agent.state.fix_load)
        for i, agent in  enumerate(world.agents):
            
            agent_load_vector = agent.state.c_load * agent.distance_manage
            load_all_cross = np.sum(agent_load_vector) 
            Plr[i] = self.PLR(load_all_cross, serv_rate[i], length_Q, agent.state.fix_load)
            if self.Q_type == "inf":
                Q_delay[i] = self.delay_Queue_inf(load_all_cross, serv_rate[i], agent.state.fix_load)
            else:
                Q_delay[i] = self.delay_Queue(load_all_cross, serv_rate[i], length_Q, Plr, agent.state.fix_load)

#        latency_agent = average_progation + Q_delay  
                  
        return Q_delay, Plr


    def delay_Queue_inf(self, load_c, servi_rate, fix_load_controller=0):
#        Queue delay
        lamda = load_c+ fix_load_controller
        mu = servi_rate
#        K = length_Q
#        rho = lamda/mu
        
        if mu > lamda:
            delay = min(35, 1/(mu-lamda))
        else:
            delay = 40
        
        return delay


    def delay_Queue(self, load_c, servi_rate, length_Q, plr, fix_load_controller=0):
#        Queue delay
        lamda = load_c+ fix_load_controller
        mu = servi_rate
        K = length_Q
        rho = lamda/mu
       
        if np.size(rho) >1:
            delay =[]
            for i, r in enumerate(rho):
                if lamda[i] == 0:
                    d = 0
                else: 
                    if r == 1:
                        d = (K-1)/(2*lamda[i])    
                    else:
                        d = (pow(r,2)+K*pow(r, K+2)-K*pow(r,K+1)-pow(r,K+2))/(lamda[i]*(1-r)*(1-pow(r,K))) 
                delay.append(d)
        else:
            if lamda == 0:
                d = 0
            else: 
                if rho == 1:
                    d = (K-1)/(2*lamda)    
                else:
                    d = (pow(rho,2)+K*pow(rho, K+2)-K*pow(rho,K+1)-pow(rho,K+2))/(lamda*(1-rho)*(1-pow(rho,K))) 
            delay=d
        return delay+1/servi_rate
#    + D_resend*plr/(1-plr)
  
    def PLR(self, load_c, servi_rate, length_Q, fix_load_controller=0):
        lamda = load_c+ fix_load_controller
        mu = servi_rate
        K = length_Q
        rho = lamda/mu
        if rho == 1:
            plr = 1/(K+1)
        else:
            plr = (1-rho)*pow(rho,K)/(1-pow(rho,K+1)) 
        return plr
         
    def observation(self, agent, world, step):  
        load_area = list(map(lambda x:x/10, agent.state.c_load))
        obs = np.concatenate(([agent.state.fix_load/10], load_area), axis=0)
        if self.Step_ob:
            obs = np.concatenate((obs, [step/self.step_num]), axis=0)
        if self.Handover:
            obs =np.concatenate((obs, agent.state.v_manage*agent.state.c_load/10), axis=0)
        if self.Que_obs:
            obs =np.concatenate((obs,[agent.state.Q_delay/40] ), axis=0)
        if self.Plr_obs:
            obs =np.concatenate((obs, [agent.state.plr]), axis=0)
        
#        if self.Step_ob and self.Handover and self.Que_obs:
#             obs = np.concatenate(([step/self.step_num],[agent.state.fix_load/10], load_area, agent.state.v_manage*agent.state.c_load/10, [agent.state.Q_delay/20]), axis=0)
#        elif (not self.Step_ob) and self.Handover and self.Que_obs:
#            obs = np.concatenate(([agent.state.fix_load/10], load_area, agent.state.v_manage*agent.state.c_load/10, [agent.state.Q_delay/20]), axis=0)
#        elif (not self.Step_ob) and  self.Handover and (not self.Que_obs):
#            obs = np.concatenate(([agent.state.fix_load/10], load_area, agent.state.v_manage*agent.state.c_load/10), axis=0)
#        elif (not self.Step_ob) and (not self.Handover) and self.Que_obs:
#            obs = np.concatenate(([agent.state.fix_load/10], load_area, [agent.state.Q_delay/20]), axis=0)
#        elif (not self.Step_ob) and (not self.Handover) and (not self.Que_obs):
#            obs = np.concatenate(([agent.state.fix_load/10], load_area), axis=0)
#        obs_standard = obs/np.max(obs)
        return obs
    #        return np.concatenate(([agent.state.fix_load], agent.state.c_load, agent.state.v_manage), axis=0)
#        
    # generate the vehicle location and load
    def generate_vehicle(self, world):
        # number of vehicles
        world.num_v = np.random.randint(50, high=120) 
        # add vehicles
        world.vehicles = [Vehicle() for i in range(world.num_v)]#        
        #vehicle location is in the coverage of agents
        select_area = np.random.randint(0,high=len(world.all_area), size = world.num_v)
        
        self.load = np.random.uniform(10, 20, size=world.num_v)
        for i, vehicle in enumerate(world.vehicles):
            x = world.all_area[select_area[i]][0]
            y = world.all_area[select_area[i]][1]
            vehicle.pos = (float(x+np.random.rand(1)),float(y+np.random.rand(1)))
#            (float((x+np.random.rand(1),x-np.random.rand(1))[x>=9]),
#                           float((y+np.random.rand(1),y-np.random.rand(1))[y>=9]))
            vehicle.load = self.load[i]/1000
            
   # get vehicle location from dataset
    def get_location_vehicle(self, world, train_step, step, dic):
        # k is episode also the data of dataset
        k = train_step%self.Group_traffic+1
#        k=1
        if  k <10:
            name = '0'+str(k)
        else:
            name = str(k) 
        with open(dic+name+'_min_10km.json','r') as f:
            diction = json.load(fp=f) 
            
        with open(dic+'RSU.json','r') as f:
            rsu_dic = json.load(fp=f)        
            

        location = diction[str(step)]
        # number of vehicles
        world.num_v = len(location)
        # add vehicles
        world.vehicles = [Vehicle() for i in range(world.num_v)]#
        # extend load to 10-20 packets/s
#        self.load = np.array(location)[:,2]
        # extend load to 5-30 packets/s
        self.load = (np.array(location)[:,2]-10)*(l_b-l_a)/10+l_a
#        np.random.uniform(10, 20, size=world.num_v)
#        load2 =  min(max(sum(self.load)/100,3),10)
#        self.load_fix_controller = (1+np.random.rand(world.agent_num)*0.3)*load2*0.1
#        
        world.load_areas = np.zeros([world.region_W, world.region_H] )
        world.vehicle_num = np.zeros([world.region_W, world.region_H] )
        
        for j, vehicle in enumerate(world.vehicles):
             x = location[j][0]/(100/world.region_W)
             y = location[j][1]/(100/world.region_H)
             vehicle.pos = (x,y)
             vehicle.load = self.load[j]/1000
             world.load_areas[int(x),int(y)] += vehicle.load 
             world.vehicle_num[int(x),int(y)]+=1
           
            
        
        for u in range(len(rsu_dic)):
            x = rsu_dic[u][0]/(100/world.region_W)
            y= rsu_dic[u][1]/(100/world.region_H)
            n = 1
#            len(world.vehicles)/800            
            load = n*((rsu_dic[u][2+step]-50)*(l_b_r-l_a_r)/(100-50)+l_a_r)
            rsu_load =min(max(l_a_r,load),l_b_r)
#            rsu.load = rsu_dic[u][3]
            world.load_areas[int(x),int(y)] += rsu_load/1000
         
            
            
          
#    #  generate the test vehicle location and load
#    def generate_vehicle_test(self, world, test_vehicles):
#        # number of vehicles
#        world.num_v = len(test_vehicles) 
#        # add vehicles
#        world.vehicles = [Vehicle() for i in range(world.num_v)]#        
#        load = [a[2] for a in test_vehicles]
#        self.load = np.array(load)
#        for i, vehicle in enumerate(world.vehicles):          
#            vehicle.pos = (test_vehicles[i][0],test_vehicles[i][1])
#            vehicle.load = test_vehicles[i][2]
            

#    # vehicle load to areas load 
#    def vehicle_to_areaload(self, world):
#        # load of each block in region_H * region_W
#        self.load_matrix = np.zeros([world.region_W, world.region_H])
#        for i, vehicle in enumerate(world.vehicles):
#            x = int(vehicle.pos[0])
#            y = int(vehicle.pos[1])
##            world.all_con_areas[x, y] > 0
#            self.load_matrix[x, y] += vehicle.load
##        self.load_matrix./=1000
    
    def test_vehicle_data(self, world,test_num):
#        vehicles = [(pos0,pos1,load)]
        vehicles = []
        for k in range(test_num):
            num_v = np.random.randint(50, high=120) 
            select_area = np.random.randint(0, high=len(world.all_area), size = num_v)
            load = np.random.uniform(0, 5, size = num_v)
            vehicle = []
            for i in range(num_v):
                x = world.all_area[select_area[i]][0]
                y = world.all_area[select_area[i]][1]
                pos_load = [float(x+np.random.rand(1)),float(y+np.random.rand(1)),load[i]]
                vehicle.append(pos_load)
            vehicles.append(vehicle)
        return vehicles
    
    
#    def get_control_path_and_delay(self, world):
#         # get the control path {'agent':{'router':[path]}}
##        world.routing_path={}
#        delay_p = []
#        for i, agent in enumerate(world.agents):
##            dists = [np.linalg.norm(world.router_pos[node] - agent.pos)/world.region_W*10 for node in list(world.topo_matrix.node)]
##            min_dists = dists.index( min(dists))
##            #find the router nearest to controller location
##            agent.router = min_dists
##            p={}
#            dela=[]
#            for x in range(world.region_W):
#                for y in range(world.region_H):
#    #            for start in list(world.topo_matrix.node):
#                    if  np.linalg.norm([x,y] - agent.pos)> agent.r :
#    #                    /world.region_W*10 
#                        #here set 1.5, because some router may be caused more distance than r
#                        #in not in coverage of agent, delay set to be inf
#                        dela.append(1000000)
#                    else:   
##                    path = nx.dijkstra_path(world.topo_matrix, start, min_dists)
##                    dis = 0
##                    for j in range(len(path)-1):                    
##                        dis += np.linalg.norm( world.router_pos[path[j+1]]- world.router_pos[path[j]])/world.region_W*10
#                        dis =  np.linalg.norm([x,y] -agent.pos)/world.region_W*10
#                        
#                        delay = (0.1 +dis*0.01/3)*2 + 1
#    #                    delay = (0.1 + len(path)*0.11 +dis*0.01/3)*2 + 1
#                        dela.append(delay)
##                    p[start] = path
#            delay_p.append(dela)
##            world.routing_path[i] =  p
#        #get delay of each wired path (agent, router)[[],[]]   
#        world.path_delay_p = delay_p 
    
    def get_delay_areas(self, world):
        world.area_delay = np.zeros([world.agent_num, world.region_W, world.region_H])
#        a1 = np.array(world.path_delay_p)
#        for i in range(world.region_W):
#            for j in range(world.region_H):
#                dists = [np.linalg.norm(world.router_pos[node] - [i,j])/world.region_W*10 for node in list(world.topo_matrix.node)]
#                min_dists = dists.index( min(dists))
#                world.area_delay[:, i, j] = a1[:, min_dists]
          
        for i, agent in enumerate(world.agents): 
            for x in range(world.region_W):
                for y in range(world.region_H):
                    if  np.linalg.norm([x,y] - np.array(agent.pos))> 1.5*agent.r :
                        #in not in coverage of agent, delay set to be inf
                        delay=1000000
                    else:   
                        dis =  np.linalg.norm([x,y] - np.array(agent.pos))/world.region_W*10
                        delay = (0.1 +dis*0.01/3)*2 + 1
    #                    delay = (0.1 + len(path)*0.11 +dis*0.01/3)*2 + 1
#                        dela=delay
                    world.area_delay[i, x, y] = delay
                
                
        
    
    def get_cross_area_prog_delay(self, world):
        
        world.distance_assign_matrix = world.all_con_areas.copy()
        for i, agent in enumerate(world.agents):
            a = np.array(agent.cross_areas)
            agent.c_area_pro_delay = world.area_delay[i,a[:,0],a[:,1]]
            # assign of the fix assignment based on distance
            dis_assi = np.zeros(len(agent.cross_areas))
            tmp = world.area_delay[:,a[:,0],a[:,1]]
            agent_in_chardge = np.argmin(tmp, axis=0)
            dis_assi[np.where(agent_in_chardge==i)]=1
            agent.distance_manage = dis_assi
   
            
        for x in range(world.region_W):
            for y in range(world.region_H):
                id_ = np.argmin( world.area_delay[:,x,y], axis=0)
                if world.distance_assign_matrix[x,y] !=0:
                    world.distance_assign_matrix[x,y] = id_+1
                LB  = (x,y) 
                for agent in world.agents:
                    distance = np.linalg.norm(np.array(agent.pos) - LB)
                    if distance <= agent.r:
                       tmp = world.area_delay[:,x,y]
                       world.agent_area_cover[agent.id,x,y] = 1
    