
import numpy as np

## Collection of reward functions 


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def objGrasped(env,thresh = 0):
    sensorData = env.data.sensordata
    return (sensorData[0]>thresh) and (sensorData[1]> thresh)

def orig_pickReward(achieved_goal, goal, objpos, env):       
    # hScale = 50
    reachDist = np.linalg.norm(goal  - achieved_goal)
    hScale = 100
    # hScale = 1000
    if env.pickCompleted and not(env.objDropped()):
        #return hScale*env.heightTarget
        return 0
    # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
    #elif (reachDist < 0.1) and (objpos[2]> (env.objHeight + 0.005)) :
     #   return -1* min(env.heightTarget, objpos[2])
    else:
        return -0.5

def general_pickReward(achieved_goal, goal, objpos, env):
    hScale = 50
    return -hScale* min(env.heightTarget, objpos[2])

def placeReward(achieved_goal, goal, objpos, env):
    # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
    placeDist = np.linalg.norm(goal  - achieved_goal)
    placeDist += np.linalg.norm(objpos - goal)
    reward = float(placeDist < 0.5) - 0.5
    return reward

def compute_reward_push(achieved_goal, goal, objpos, env= None):
    c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
    assert achieved_goal.shape == goal.shape
    assert objpos.shape == goal.shape
    reachDist = np.linalg.norm(goal  - achieved_goal, axis=-1)
    pushDist = np.linalg.norm(np.array([i[:2] for i in objpos]) \
                               - np.array([i[:2] for i in goal]), axis=-1)
    reachRew = reachDist
    pushRew = pushDist
#    if reachDist < 0.05:
#        pushRew = -pushDist
#        #pushRew = 1000*(env.maxPushDist - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
#        #pushRew = max(pushRew, 0)
#    else:
#        pushRew = 0
    reward = reachRew + pushDist
    return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None]

def compute_reward_pick_place(achieved_goal, goal, objpos, env):
    r1 = orig_pickReward(achieved_goal, goal, objpos, env)
    r2 = placeReward(achieved_goal, goal, objpos, env)
    return np.floor(r1+r2)

def compute_reward(achieved_goal, goal, objpos=None, task = 'reach', env = None):
    if task == 'reach':
        reward = goal_distance(achieved_goal, goal)  
        #print([0 if i<0.05 else -1 for i in reward])                                    
        return [0 if i<0.05 else -1 for i in reward]
    elif task == 'push':
        #rewards = [compute_reward_push(achieved_goal, goal, objpos, env)[0] for  action, ob in zip(achieved_goal, goal)]
        reward = compute_reward_push(achieved_goal, goal, objpos, env)[0]
        #return np.array(rewards)
        return [0 if i<0.4 else -1 for i in reward]
    elif task == 'pick_place':
        rewards = [compute_reward_pick_place(achieved_goal, goal, objpos, env) for  action, ob in zip(achieved_goal, goal)]
        #print(rewards)
        return rewards

def compute_reward_pure(achieved_goal, goal, objpos=None, task = 'reach', env = None):
    if task == 'reach':
        reward = goal_distance(achieved_goal, goal)                                      
        return [1/i for i in reward]
    elif task == 'push':
        #rewards = [compute_reward_push(achieved_goal, goal, objpos, env)[0] for  action, ob in zip(achieved_goal, goal)]
        reward = compute_reward_push(achieved_goal, goal, objpos, env)[0]
        #return np.array(rewards)
        return [1/i for i in reward]
    elif task == 'pick_place':
        rewards = [compute_reward_pick_place(achieved_goal, goal, objpos, env) for  action, ob in zip(achieved_goal, goal)]
        return [1/(-1*i+1) for i in rewards]

