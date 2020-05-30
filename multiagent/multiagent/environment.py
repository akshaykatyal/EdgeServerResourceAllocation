import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multiagent.multi_discrete import MultiDiscrete

length_Q = 200
# D_resend = 20

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, arglist, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.Q_type = arglist.Q_type
        # environment parameters
#        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
#        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
#        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward, share reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        
        
        self.step_num_max = 23

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []      
                
            # action space ????????????????????????????
#            for i in range(agent.action.dim_a):
            u_action_space = spaces.Discrete(agent.action.dim_a)
    #            u_action_space = spaces.Box(low=0.0, high=1.0, shape=(agent.action.dim_a,), dtype=np.float32)
            total_action_space.append(u_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
                
            # observation space
            obs_dim = len(observation_callback(agent, self.world, step=0))
            self.observation_space.append(spaces.Box(low=0, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
#            agent.action.c = np.zeros(self.world.dim_c)

#        # rendering
#        self.shared_viewer = shared_viewer
#        if self.shared_viewer:
#            self.viewers = [None]
#        else:
#            self.viewers = [None] * self.n
#        self._reset_render()

    def step(self, action_n, step_num, l_type ='maddpg'):
        #action_n: probility of cross areas (agent,cross_areas)
        obs_n = []
#        reward_n = []
        done_n = []
        info_n = {'n': []}
#        delay_opt = []
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step(action_n)
        
        
        # get the delay before taken action
       
#        Q_delay = np.zeros(self.world.agent_num)
#
#        for agent in self.agents:
#            # get Q delay before taken action
#            Q_delay[agent.id] = self.get_latency_one_agent(agent, self.world, agent.serve_rate)
#            agent.state.Q_delay = Q_delay[agent.id]
#
#        self.world.last_delay = self.get_latency_area(self.world, Q_delay, self.world.all_con_areas)
#        

        # record observation for each agent
        for agent in self.agents:

            obs = self._get_obs(agent, step_num)
#            obs_max = max(np.max(obs),obs_max)
            obs_n.append(obs)
#            obs_n.append(np.hstack((obs, agent.state.v_manage)))
#            reward_n.append(self._get_reward(agent, Flag_areas))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))
        
#        delay_opt.append(self.world.delay_opt)

        # all agents get total reward in cooperative case
#        reward = np.sum(reward_n)
#        if Flag_areas:
#            delay_ma_areas, fix_delay_areas= self._get_reward(agent, Flag_areas)
#            return delay_ma_areas, fix_delay_areas
#        else:
        reward, changed_number, changed_area_num, delay_in_area, reward_real, plr= self._get_reward(l_type)            
        if self.shared_reward and np.size(reward)==1:
            reward_n = [reward] * self.n
        else:
            #reward of agents
            reward_n = reward
            
#        get_average_delay(action_n)
        obs_stand_n = obs_n
#            [c/obs_max for c in obs_n]
        
        return obs_stand_n, reward_n, done_n, info_n, changed_number, changed_area_num, delay_in_area, reward_real, plr

    def reset(self, train_step, step, arglist, IF_test=False, TEST_V = None):
        # reset world
        self.reset_callback(self.world, train_step, step, arglist, IF_test, TEST_V)
        # reset renderer
#        self._reset_renxder()
        # record observations for each agent
        obs_n = []
#        obs_max = 0
        self.agents = self.world.policy_agents
        
        Q_delay = np.zeros(self.world.agent_num)

        for agent in self.agents:
            # get Q delay before taken action
           
            Q_delay[agent.id], agent.state.plr = self.get_latency_one_agent(agent, self.world, agent.serve_rate)
            agent.state.Q_delay = Q_delay[agent.id]
            obs = self._get_obs(agent, step)
            obs_n.append(obs)
#            np.hstack((obs, agent.state.v_manage)))
#            obs_n.append(agent.state.v_manage)
            #归一化，可能有点问题
#            obs_max = max(np.max(obs),obs_max)
#        obs_max =1
#        obs_stand_n = [c/obs_max for c in obs_n]
        self.world.last_delay = self.get_latency_area(self.world, Q_delay, self.world.all_con_areas)
        
        return obs_n
    
    
    def get_latency_area(self, world, Q_delay, agent):
        # 获取延时pro+area
        delay_in_area = np.zeros([world.region_W, world.region_H])
        for x in range(world.region_W):
            for y in range(world.region_H): 
                if agent[x,y]>0:
                    i = int(agent[x,y])-1
                    delay_in_area[x,y] = Q_delay[i] + world.area_delay[i,x,y]
        return delay_in_area
    
    
    # Que delay of agent
    def get_latency_one_agent(self, agent, world, serv_rate):
        # average latency of agent based on Small areas
        agent_load_vector = agent.state.c_load * agent.state.v_manage
        load_all_cross = np.sum(agent_load_vector)
        if self.Q_type == "inf":
            Q_delay = self.delay_Queue_inf(load_all_cross, serv_rate, agent.state.fix_load)   
            plr = 0 
        else:
            plr = self.PLR(load_all_cross, serv_rate, length_Q, agent.state.fix_load)
            Q_delay = self.delay_Queue(load_all_cross, serv_rate, length_Q, plr, agent.state.fix_load)      
        return Q_delay, plr
    
    def delay_Queue_inf(self, load_c, servi_rate, fix_load_controller=0):
#        Queue delay
        lamda = load_c+ fix_load_controller
        mu = servi_rate
#        K = length_Q
#        rho = lamda/mu
        
        if mu > lamda:
            delay = min(40, 1/(mu-lamda))
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
    
    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent, step):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world, step)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return 0
#        False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, l_type):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(self.world, l_type)
    # simple_controllr -> reward()

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
#        agent.action.p_ctl = np.zeros(agent.action.dim_a)        
        agent.action.p_ctl = action
        
#        
#    # reset rendering assets
#    def _reset_render(self):
#        self.render_geoms = None
#        self.render_geoms_xform = None

#    # render environment
#    def render(self, mode='human'):
#        from multiagent import rendering
#        if mode == 'human':
#            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#            message = ''
#            for agent in self.world.agents:
#                for other in self.world.agents:
#                    if other is agent: continue
#                    if np.all(other.state.c == 0):
#                        word = '_'
#                    else:
#                        word = alphabet[np.argmax(other.state.c)]
#                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
#            print(message)
#
#        for i in range(len(self.viewers)):
#            # create viewers (if necessary)
#            if self.viewers[i] is None:
#                # import rendering only if we need it (and don't import for headless machines)
#                #from gym.envs.classic_control import rendering
#                
#                self.viewers[i] = rendering.Viewer(700,700)
#
#        # create rendering geometry
#        if self.render_geoms is None:
#            # import rendering only if we need it (and don't import for headless machines)
#            #from gym.envs.classic_control import rendering
#            from multiagent import rendering
#            self.render_geoms = []
#            self.render_geoms_xform = []
#            for entity in self.world.entities:
#                geom = rendering.make_circle(entity.size)
#                xform = rendering.Transform()
#                if 'agent' in entity.name:
#                    geom.set_color(*entity.color, alpha=0.5)
#                else:
#                    geom.set_color(*entity.color)
#                geom.add_attr(xform)
#                self.render_geoms.append(geom)
#                self.render_geoms_xform.append(xform)
#
#            # add geoms to viewer
#            for viewer in self.viewers:
#                viewer.geoms = []
#                for geom in self.render_geoms:
#                    viewer.add_geom(geom)
#
#        results = []
#        for i in range(len(self.viewers)):
#            
#            # update bounds to center around agent
#            cam_range = 1
#            if self.shared_viewer:
#                pos = np.zeros(self.world.dim_p)
#            else:
#                pos = self.agents[i].state.p_pos
#            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
#            # update geometry positions
#            for e, entity in enumerate(self.world.entities):
#                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
#            # render to display or array
#            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))
#
#        return results
    
    

#    # create receptor field locations in local coordinate frame
#    def _make_receptor_locations(self, agent):
#        receptor_type = 'polar'
#        range_min = 0.05 * 2.0
#        range_max = 1.00
#        dx = []
#        # circular receptive field
#        if receptor_type == 'polar':
#            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
#                for distance in np.linspace(range_min, range_max, 3):
#                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
#            # add origin
#            dx.append(np.array([0.0, 0.0]))
#        # grid receptive field
#        if receptor_type == 'grid':
#            for x in np.linspace(-range_max, +range_max, 5):
#                for y in np.linspace(-range_max, +range_max, 5):
#                    dx.append(np.array([x,y]))
#        return dx


# vectorized wrapper for a batch of multi-agent environments
## assumes all environments have the same observation and action space
#class BatchMultiAgentEnv(gym.Env):
#    metadata = {
#        'runtime.vectorized': True,
#        'render.modes' : ['human', 'rgb_array']
#    }
#
#    def __init__(self, env_batch):
#        self.env_batch = env_batch
#
#    @property
#    def n(self):
#        return np.sum([env.n for env in self.env_batch])
#
#    @property
#    def action_space(self):
#        return self.env_batch[0].action_space
#
#    @property
#    def observation_space(self):
#        return self.env_batch[0].observation_space
#
#    def step(self, action_n, time):
#        obs_n = []
#        reward_n = []
#        done_n = []
#        info_n = {'n': []}
#        i = 0
#        for env in self.env_batch:
#            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
#            i += env.n
#            obs_n += obs
#            # reward = [r / len(self.env_batch) for r in reward]
#            reward_n += reward
#            done_n += done
#        return obs_n, reward_n, done_n, info_n
#
#    def reset(self):
#        obs_n = []
#        for env in self.env_batch:
#            obs_n += env.reset()
#        return obs_n
#
#    # render environment
#    def render(self, mode='human', close=True):
#        results_n = []
#        for env in self.env_batch:
#            results_n += env.render(mode, close)
#        return results_n
