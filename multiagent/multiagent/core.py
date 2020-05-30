import numpy as np
import networkx as nx


size_map = 1/2
# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # controlled vehicles
        self.v_manage = None
        # the un-changed part of load
        self.fix_load = 0
        # the load of crossover areas
        self.c_load = None
        # load of agent's areas
        self.load =None 
        # average delay of agents and its controlled areas
        self.avg_delay = 0
        # Queue delay
        self.Q_delay = 0
        # Queue packet loss rate
        self.plr = 0
        

# action of the agent
class Action(object):
    def __init__(self):     
        # probaility of control
        self.p_ctl = None
        # action dimensionality
        self.dim_a = 0
        # real action of agent
        self.v_manage = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # ID
        self.id = 0
        # name 
        self.name = ''
        # entity can move / be pushed
        self.movable = False
#        # color
#        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        
class Vehicle(Entity):
     def __init__(self):
         super(Vehicle, self).__init__()
         self.pos = None
         self.load = 0

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None        
        # location of agents
        self.pos = None
        # number of cross-over areas
        self.cross_over_num = None
        # the (x-y) areas under control coverage
        self.areas = []
        # coverage radius
        self.r = int(4*size_map)
        # crossover areas of control
        self.cross_areas =[]
#        # ID of areas under control coverage
#        self.areas_id = []
        self.distance =[]
        # serve rate of agents 1 packets/ms
        self.serve_rate = 10
        # the router connected 
        self.router = 0
        # manage based on distance
        self.distance_manage = None


# properties of agent entities
class Agent_Local(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # location of agents
        self.pos = None
        # the (x-y) areas under control coverage
        self.areas = []
        # coverage radius
        self.r = int(0.5*size_map)
        # crossover areas of control
        self.cross_areas =[]
        self.distance =[]
        # serve rate of agents 1 packets/ms
        self.serve_rate = 10
        # manage based on distance
        self.distance_manage = None

class Router(object):
    def __init__(self):
        super(Agent, self).__init__()
        self.pos = None
        
#class Small_area(object):
     

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.vehicles = []
        # communication channel dimensionality
        self.dim_c = 0
        #number of vehicles
        self.num_v = 50
        self.agent_num = 4
        self.agent_num_local = 2
        # region distance 10km and divide into 20*20 pieces
        self.region_W = int(size_map*10)
        self.region_H = int(size_map*10)
        
        self.centralized_Delay=0

        #all the areas in all agents
        self.all_area = []      
        # for each of SAs who in charge [region_W,region_H]
        self.area_management= None

        # all the cross over areas ID
        self.cross_a = []
        # multi controlled areas (agent->area ID)
        self.mul_area_c = {}
        # sending rate of communicatin 3*10^8m/s
        self.sending_rate = 300
        # all controled areas 
        self.all_con_areas = np.zeros([self.region_W, self.region_H] )
        self.last_all_con_areas =None
        
        
        # opti assingment based on progam latency
        self.delay_opt = 0
        self.delay_ma = np.zeros(self.agent_num)
        # the maddpg assignment (agents, vehicles) of all vehicles
        self.MA_v_c = None
        # last step assignment
        self.MA_v_c_last = None
        # fix management 
        self.Fix_v_c = None
        # controller can management vehicles (agents, vehicles)=1 if it can
        self.v_c_possible = None
        
        # coverage of rsu 0.5km
        self.rsu_r = 0.5
        #number of vehicle in each areas
        self.vehicle_num = None
#       
#        # Set RSU and router topo
#        topology = '/Users/eva/Documents/GitHub/topo/india35.matrix'
#        self.topo_matrix = nx.Graph(np.loadtxt(topology, dtype=int))
#         # number of routers
#        self.num_router = self.topo_matrix.number_of_nodes()
#        
##        po = []
##        with open('/Users/eva/Documents/GitHub/topo/India_35_nodes.txt', 'r') as f:
##            for j in f.readlines():
##              po.append( list(map(lambda x: float(x), j[:-1].split(' ')[2:4])))
##      
##        self.router_pos = np.array(po)/(50/self.region_W)
        self.topo_matrix = None
        self.num_router = 0
        self.router_pos = None
        self.routing_path = None
        self.path_delay_p = None
        #small area delay
        self.area_delay=None
        
        self.load_areas = np.zeros([self.region_W, self.region_H] )
        
#        for atitude and longitude change
        self.offsize_lat = 22.9427
        self.offsize_lng = 43.3175
        self.zom = 10.256
        
        self.distance_assign_matrix = np.zeros([self.region_W, self.region_H] )
        
        self.agent_area_cover = np.zeros([self.agent_num, self.region_W, self.region_H] )
        
        self.delay_in_area_fix = None
             

    def set_topo(self, dic):
        # Set RSU and router topo

        topology = dic+'Rio_de_Janeiro.matrix'
        self.topo_matrix = nx.Graph(np.loadtxt(topology))
#        , dtype=int
         # number of routers
        self.num_router = self.topo_matrix.number_of_nodes()

        po = []
        with open(dic+'Rio_de_Janeiro_nodes.txt', 'r') as f:
            for j in f.readlines():
              po.append( list(map(lambda x: float(x), j[:-1].split(' ')[1:3]))) 
        po_array = np.array(po)
        
        po_array[:,0] = (po_array[:,0]+self.offsize_lat)*self.zom*self.region_W
        po_array[:,1] =(po_array[:,1]+self.offsize_lng)*self.zom*self.region_H
        self.router_pos = po_array
#        /(50/self.region_W)
    
    def set_action_dim(self):
        for agent in self.agents:      
            agent.action.dim_a = len(self.mul_area_c.get(agent.id)[0])
        

       
    # set server rate of agents 1 packets/ms
    def set_server_rate(self):
        for agent in self.agents:
            agent.serve_rate = 10
#            packages/ms
#            np.random.randint(2, high = 6)
    
    # get agent -> crossover areas (mul_area_c)&  agent->all areas (areas)
    def get_crossover(self):
        cross_area = []
        all_area = []
        self.cross_areas_num = 0
        for x in range(self.region_H):
            for y in range(self.region_W):
                # the left and bottom point
                LB  = (x,y) 
                _term = 0
                for agent in self.agents:
                    distance = np.linalg.norm(np.array(agent.pos) - LB)
                    if distance <= agent.r:  
                        agent.areas.append(LB)
                        agent.distance.append(distance)
                        _term += 1
                        if _term == 2:
                            cross_area.append(LB)
                            # all areas
                        if _term == 1:
                            all_area.append(LB)

                
        self.cross_areas_num = len(cross_area)
        self.cross_a = cross_area
        self.all_area = all_area
  
        for agent in self.agents:  
            listA = agent.areas
            id_agent = list(set(listA).intersection(set(cross_area)))
            id_agent.sort()
            self.mul_area_c.setdefault(agent.id,[]).append(id_agent)
            agent.cross_areas = id_agent.copy()
            if list(set(agent.cross_areas).difference(set(self.cross_a))) != [] :
                print("error in agent cross areas.")
                input()
             # set region fix areas from 1:agent_num
            for value in agent.areas:
                self.all_con_areas[value[0],value[1]] = agent.id+1
#                fix_manage_areas(self, agent,)
#    

       
        
                
    # multi agents action -> real action           
    def get_real_action(self, a_probality):
        
        agent_areas_matrix = np.ones([self.agent_num, self.cross_areas_num])*(-10)
        for i, agent in enumerate(self.agents):
            for value in self.mul_area_c[agent.id]:
                for k in range(len(value)):
                    p = self.cross_a.index(value[k])
                    if a_probality[i][k] in agent_areas_matrix[:i,p]:
                        agent_areas_matrix[i,p] = a_probality[i][k]+np.random.uniform(low=-0.1, high=0.1, size=1)
                        if agent_areas_matrix[i,p] in agent_areas_matrix[:i,p]:
                            agent_areas_matrix[i,p] = a_probality[i][k]+np.random.uniform(low=0, high=0.1, size=1)
                    else:
                        agent_areas_matrix[i,p] = a_probality[i][k]   
        
        self.last_all_con_areas = self.all_con_areas.copy()
        agent_max = np.where(agent_areas_matrix==np.max(agent_areas_matrix, axis=0))
        te = np.array(agent_max)
        agent_max = te[:,te[1].argsort()][0]
        k =0
        # set region cross over areas from 1:agent_num
        for value in te[1]:
            self.all_con_areas[self.cross_a[value][0], self.cross_a[value][1]] = te[0][k]+1
            k+=1
        if len(agent_max) > len(self.cross_a):
            print(agent_max)
        return  agent_max   

    
    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self, action_n):
        # agents control the areas (cross_areas_region,1)
        real_action = self.get_real_action(action_n) 
        if len(real_action) > len(self.cross_a):
            print("error")
            print(action_n)
            print(real_action)
        # update agent state
        for agent in self.agents:
            # global area id
            area_id = np.argwhere(real_action==agent.id)
            # update agent.state.v_manage
            self.update_agent_state(agent, area_id)  
        
            
    def update_agent_state(self, agent, real_action):
#        # set communication state (directly for now) 
        new_v_manage = np.zeros(len(agent.state.v_manage))
    
        if real_action is not None and real_action.size>0:
            ra = np.hstack(real_action).tolist()
            if max(ra) >= len(self.cross_a):
                print("error in ra!!")
            for i in ra:
                node = self.cross_a[i]
                te = agent.cross_areas
                if node not in te:
                    print("error in node!!")
                index_ = te.index(node)
                new_v_manage[index_]=1
            
        agent.state.v_manage = new_v_manage
        agent.action.v_manage = new_v_manage


    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt



