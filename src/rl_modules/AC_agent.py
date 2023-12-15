import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import matplotlib.pyplot as plt
from rewards import *

"""
Classes:
    - ddpg_agent: Main class for the DDPG agent.
        - __init__: Initializes the agent, networks, and other components including HyperNet.
        - learn: Main method to train the agent using both its own networks and HyperNet.
        - _preproc_inputs: Preprocesses inputs for the network.
        - _select_actions: Selects actions based on current policy.
        - _update_normalizer: Updates observation and goal normalizers.
        - _soft_update_target_network: Softly updates the target networks.
        - _update_network: Updates the actor and critic networks using HyperNet for advanced policy updates.
        - _eval_agent: Evaluates the agent's performance.
        - load_model: Loads model weights and normalizer stats.

Functions:
    - plot_successrate: Plots the success rate over epochs.
    - plot_returntask: Plots the average return over epochs.

HyperNet Usage:
    The HyperNet plays a dual role in this implementation. Firstly, it enhances the actor network's policy learning, providing an advanced approach to policy optimization. Secondly, it is utilized for transfer learning purposes. By leveraging the pre-trained models in HyperNet, the AC agent can effectively transfer knowledge from previous tasks or environments, thereby improving learning efficiency and adaptation in new, but similar, tasks or environments.

"""

def plot_successrate(success_rate, filename2, x=None, window=5):

    M = len(success_rate)
    plt.figure(num=1)
    if x is None:
        x = [i for i in range(M)]
    plt.ylabel('Success Rate')
    plt.xlabel('Epoch')
    plt.plot(x, success_rate)
    plt.savefig(filename2)
    np.save(filename2, success_rate)

def plot_returntask(success_rate, filename2, x=None, window=5):

    M = len(success_rate)
    plt.figure(num=2)
    if x is None:
        x = [i for i in range(M)]
    plt.ylabel('Avg Return')
    plt.xlabel('Epoch')
    plt.plot(x, success_rate)
    plt.savefig(filename2)
    np.save(filename2, success_rate)


"""
ddpg with HER (MPI-version)

"""
filename = 'Results\push-v1'
#goal = np.array([0.1 , 0.8 , 0.02])

class ddpg_agent:
    def __init__(self, args, env, env_params, modules,HyperNet):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.task_type = env.task_type
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        #self.actor_network_mod1 = actor_module_init(env_params)
        #self.critic_network_mod1 = critic_module_init(env_params)
        #self.actor_network_mod2 = actor_module_mid(env_params)
        #self.critic_network_mod2 = critic_module_mid(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(HyperNet.actor_network)
        sync_networks(self.critic_network)
        #sync_networks(self.actor_network_mod1)
        #sync_networks(self.critic_network_mod1)
        #sync_networks(self.actor_network_mod2)
        #sync_networks(self.critic_network_mod2)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.exp_time = str(datetime.now())+'_'+self.task_type+'/'
        self.exp_time = self.exp_time.replace("-","_")
        self.exp_time = self.exp_time.replace(":","_")
        self.exp_time = self.exp_time.replace(".","_")
        # if use gpu
        if self.args.cuda:
            HyperNet.actor_network.cuda()
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k,self.env, compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if self.args.ismodular == True:
            self.load_model('module_models/'+modules[0]+'.pt')
            self.exp_path = self.args.exp_dirm
        else:
            self.exp_path = self.args.exp_dir
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.exp_path):
                os.mkdir(self.exp_path)
            self.exp_path = os.path.join(self.exp_path, self.exp_time)
            if not os.path.exists(self.exp_path):
                os.mkdir(self.exp_path)            
            self.model_path = os.path.join(self.exp_path, self.args.save_dir)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            self.fig_path = os.path.join(self.exp_path, self.args.fig_dir)
            if not os.path.exists(self.fig_path):
                os.mkdir(self.fig_path)
            self.res_path = os.path.join(self.exp_path, self.args.res_dir)
            if not os.path.exists(self.res_path):
                os.mkdir(self.res_path)


    def learn(self, task,HyperNet):
        """
        train the network

        """
        # start to collect samples
        success_rate_history = []
        return_rate_history = []
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions, mb_maxReachDist, mb_objpos = [], [], [], [], [],[]
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions, ep_maxReachDist, ep_objpos = [], [], [], [], [],[]
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation[:3]
                    ag = observation[:3]
                    #ag = self.env.get_fingerCOM()   ######reach task
                    g = observation[6:9]
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            #input_tensor = self._preproc_inputs(ag, input_tensor)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, reward, _, info = self.env.step(action)
                        #print(reward)
                        #ag_new = self.env.get_fingerCOM()     ######reach task
                        ag_new = observation_new[:3]
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = observation_new[:3]
                        ag = ag_new
                        maxReachDist = self.env.maxReachDist
                        ep_maxReachDist.append(maxReachDist.copy())
                        ep_objpos.append(observation_new[3:6])
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_maxReachDist = np.reshape(ep_maxReachDist, [-1, 1])
                    ep_objpos.append(observation_new[3:6])
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_maxReachDist.append(ep_maxReachDist)
                    mb_objpos.append(ep_objpos)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb_maxReachDist = np.array(mb_maxReachDist)
                mb_handpos = np.array(mb_objpos)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_maxReachDist,mb_handpos])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions, mb_maxReachDist,mb_handpos])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network(HyperNet)

                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            [success_count,return_count] = self._eval_agent()
            success_rate_history.append(success_count)
            return_rate_history.append(return_count)
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] task: {}, epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), task, epoch, success_rate_history[epoch]))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                           self.model_path + task + '.pt')
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, HyperNet.actor_network.state_dict(), self.critic_network.state_dict()], \
                           'module_models/multitask' + '.pt')
            if((epoch+1) % 10 ==0):
                plot_successrate(success_rate_history, self.fig_path+'succ_'  + task, x=None, window=5)
                np.save(self.res_path+'succ_'  + task, success_rate_history)
            if((epoch+1) % 10 ==0):
                plot_returntask(return_rate_history, self.fig_path+'ret_'  + task, x=None, window=5)
                np.save(self.res_path+'ret_'  + task, return_rate_history)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, mb_maxReachDist, mb_objpos = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       'objpos':mb_objpos
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self,HyperNet):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        actions_real2 = HyperNet.actor_network(inputs_norm_tensor)
        actor_loss2 = -self.critic_network(inputs_norm_tensor, actions_real2).mean()
        actor_loss2 += self.args.action_l2 * (actions_real2 / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        HyperNet.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_loss2.backward()
        sync_grads(self.actor_network)
        sync_grads(HyperNet.actor_network)
        self.actor_optim.step()
        HyperNet.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        total_return_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            per_return_rate = []
            observation = self.env.reset()
            obs = observation[:3]
            g = observation[6:9]
            objpos = observation[3:6]
            for _ in range(300):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new[:3]
                g = observation[6:9]
                #success = compute_reward(goal, obs[:3], _) + 1
                #success = info['success']
                succ = returnval = compute_reward(np.array([obs]),np.array([g]),objpos=np.array([objpos]),task=self.task_type,env=self.env)
                success = float(succ[0]==0)
                returnval = compute_reward_pure(np.array([obs]),np.array([g]),objpos=np.array([objpos]),task=self.task_type,env=self.env)
                per_success_rate.append(success)
                per_return_rate.append(returnval[0])
                #print(success)
            total_success_rate.append(per_success_rate)
            total_return_rate.append(per_return_rate)
        total_success_rate = np.array(total_success_rate)
        total_return_rate = np.array(total_return_rate)
        local_return_rate = np.mean(total_return_rate[:, -1])
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_return_rate = MPI.COMM_WORLD.allreduce(local_return_rate, op=MPI.SUM)
        return [global_success_rate / MPI.COMM_WORLD.Get_size(),global_return_rate / MPI.COMM_WORLD.Get_size()]

    def load_model(self, model_path):
        o_mean, o_std, g_mean, g_std, model, model1 = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor_network.load_state_dict(model)
        self.critic_network.load_state_dict(model1)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
        self.g_norm.mean = g_mean
        self.g_norm.std = g_std

    def load_model_mod1(self, model_path):
        o_mean, o_std, g_mean, g_std, model, model1 = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor_network_mod1.load_state_dict(model)
        self.critic_network_mod1.load_state_dict(model1)
        self.actor_target_network.load_state_dict(self.actor_network_mod1.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network_mod1.state_dict())
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
        self.g_norm.mean = g_mean
        self.g_norm.std = g_std
        
    def load_model_mod2(self, model_path):
        o_mean, o_std, g_mean, g_std, model, model1 = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.actor_network_mod2.load_state_dict(model)
        self.critic_network_mod2.load_state_dict(model1)
        self.actor_target_network.load_state_dict(self.actor_network_mod2.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network_mod2.state_dict())
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
        self.g_norm.mean = g_mean
        self.g_norm.std = g_std
