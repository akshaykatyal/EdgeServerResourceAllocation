import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
l_a = 5 # packets/ms, lower-bound of arriving rate of vehicles of G1
l_b = 50 #upper bound of packets for Group1
l_a_r = 100 # packets/ms of edge server
l_b_r = 200# packets of Q size
remote_delay_max = 8 #ms
Qlength = 200 #for the server
#transmission delay for the edge server
Tdelay=l_a/l_a_r
#setting Q length
Qlength= 200

# environment for all agents in the multiagent resourece allocattion environment
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
        self.n = len(world.policy_agents)
        # resource allocate scenario callback
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.Q_type = arglist.Q_type
        #getting the reward from the user
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        #defining maximum number of step for the environment
        self.max_step = 8
        # configure spaces
        self.action_space = []
        #the observation space
        self.observation_space = []
        for agent in self.agents:
            #defining total action space for agents
            total_action_space = []
            #vehicle agent action space
            u_action_space = spaces.Discrete(agent.action.dim_a)
            #total action space available for agaent
            total_action_space.append(u_action_space)
            # total action space
            if len(total_action_space) > 1:
                # the action spaces for the vehicles are discrete
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
                
            # observation space for the environment
            obs_dim = len(observation_callback(agent, self.world, step=0))
            self.observation_space.append(spaces.Box(low=0, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
#function for the agent taking steps in environment
    def step(self, action_n, step_num, learning_type ='maddpg'):
        #obseration space for the steps for vehicle agents
        obs_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # setting actions for agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step(action_n)
        for agent in self.agents:
#getting observation for the agents in environment
            obs = self._get_obs(agent, step_num)
            obs_n.append(obs)
#getting the done for the steps agent take
            done_n.append(self._get_done(agent))
#getting information for each step
            info_n['n'].append(self._get_info(agent))
        reward, changed_number, changed_group_num, delay_in_group, reward_real= self._get_reward(learning_type)
        if self.shared_reward and np.size(reward)==1:
            reward_n = [reward] * self.n
        else:
            #if no reward is received by the agent
            reward_n = reward
        obser=obs_n
        return obser, reward_n, done_n, info_n, changed_number, changed_group_num, delay_in_group, reward_real

#function to reset the environment
    def reset(self, train_step, step, arglist, IF_test=False, TEST_V = None):
        # reset environment world
        self.reset_callback(self.world, train_step, step, arglist, IF_test, TEST_V)
        obs_n = []
        self.agents = self.world.policy_agents
        Q_delay = np.zeros(self.world.agent_num)+Tdelay
        for agent in self.agents:
            # get Q delay
            Q_delay[agent.id] = self.get_latency_one_agent(agent, self.world, agent.service_rate)
            agent.state.Q_delay = Q_delay[agent.id]
            obs = self._get_obs(agent, step)
            #appending the number of observations
            obs_n.append(obs)
        self.world.last_delay = self.get_latency_group(self.world, Q_delay, self.world.all_con_group)
        #function returns the number of observations
        return obs_n
    
    def get_latency_group(self, world, Q_delay, agent):
#delay in the group in resource allocation
        delay_in_group = np.zeros([world.region_W, world.region_H])
        for x in range(world.region_W):
            for y in range(world.region_H): 
                if agent[x,y]>0:
                    i = int(agent[x,y])-1
                    delay_in_group[x,y] = Q_delay[i] + world.group_delay[i,x,y]
        #the function returnthe delay in the group
        return delay_in_group
#function getting latency of one agent
    def get_latency_one_agent(self, agent, world, service_rate):
        #  latency of agent based on Small group

        agent_load_vector = agent.state.c_load * agent.state.v_manage
        load_all_cross = np.sum(agent_load_vector)
        # getting the value of q delay
        if self.Q_type == "inf":
            Q_delay = self.delay_Queue_inf(load_all_cross, service_rate, agent.state.fix_load)

        else:
            Q_delay = self.delay_Queue(load_all_cross, service_rate, Qlength, agent.state.fix_load)
        return Q_delay + Tdelay

    # function to get q delay
    def delay_Queue_inf(self, load_server, service_rate, fix_load_server=0):

        lamda = load_server + fix_load_server
        mu = service_rate
        K = Qlength
        rho = lamda / mu
        if mu > lamda:
            delay = min(40, 1 / (mu - lamda))
        else:
            delay = 50
        #returns the delay q for simulation
        return delay

    # function to get the value of q delay
    def delay_Queue(self, load_server, service_rate, Qlength, fix_load_server=0):

        lamda = load_server + fix_load_server
        mu = service_rate
        K = Qlength
        rho = lamda / mu
        Qlen = 200
        if np.size(rho) > 1:
            delay = []
            for i, r in enumerate(rho):
                if lamda[i] == 0:
                    d = 0
                else:
                    if r == 1:
                        d = Qlen / mu + (K - 1) / (2 * lamda[i])
                    else:
                        d = Qlen / mu + (pow(r, 2) + K * pow(r, K + 2) - K * pow(r, K + 1) - pow(r, K + 2)) / (
                                    lamda[i] * (1 - r) * (1 - pow(r, K)))
                delay.append(d)
        else:
            if lamda == 0:
                d = 0
            else:
                if rho == 1:
                    d = Qlen / mu + (K - 1) / (2 * lamda)
                else:
                    d = (pow(rho, 2) + K * pow(rho, K + 2) - K * pow(rho, K + 1) - pow(rho, K + 2)) / (
                                lamda * (1 - rho) * (1 - pow(rho, K)))
            delay = d
        return delay + Tdelay

    
    # get info used for the results
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation of a particular agent in the environment
    def _get_obs(self, agent, step):
        if self.observation_callback is None:
            return np.zeros(0)
        #returns the observation for agent
        return self.observation_callback(agent, self.world, step)

    # get dones for a agent in a environment
    def _get_done(self, agent):
        if self.done_callback is None:
            return 0
        #returns the number of dones for a agent
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, learning_type):
        if self.reward_callback is None:
            return 0.0
        #returns the reward for the type of lerning algorithm
        return self.reward_callback(self.world, learning_type)
    # simple_controllr -> reward()

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
#getting the probability of action from agent
        agent.action.p_ctl = action

