import cvxpy as cp
import numpy as np

def opt_assignment(number_vehicle, num_agent, delay_p, load_v, lamda):
    # Problem data.
    # delay_p: propagate delay of vehicles delay_p (vehicles,agents)
    # load_v: load of vehicles(vehicles,1)
    # lamda: the service rate of agents

    
    # Construct the problem.
    # the assignment varaiables
    x = cp.Variable((num_agent,number_vehicle))
    
    # manage delay of controller (load)(1,agent)
    delay_m = x@(load_v/lamda)

    # one vehicle under one controller
    B = np.ones([1,num_agent])
    b = np.ones([1,number_vehicle])
    
    # sum of trace
    delay_p_new =cp.trace(delay_p@x)
    # manage delay (1,v)
    delay_m_new = cp.sum(delay_m.T@x)
 
    
#    objective = cp.Minimize((delay_p_new+ delay_m_new)/number_vehicle)
    objective = cp.Minimize((delay_p_new)/number_vehicle)
    constraints = [0 <= x, x <= 1, B@x == b]
    prob = cp.Problem(objective, constraints)
    
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    
    
    x_value = x.value
    x_result = np.around(x_value)
    #.astype(int)
#    print(x_result)
#    print(result)
    return x_result
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    #print(constraints[0].dual_value)
    
if __name__ == '__main__':
    
    number_vehicle = 5
    delay_p = np.array([[2,3,4,2,1],[1,4,6,1,2],[3,2,2,5,4]]).T
    load_v = np.array([2,3,4,1,2]).reshape(number_vehicle,1)
    lamda = np.array([2,2,4])
    num_agnet = 3
    
    x_result = opt_assignment(number_vehicle, num_agnet, delay_p, load_v, lamda)