from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv
env = SawyerReachPushPickPlaceEnv()
env.set_parameters(obs_type='with_goal', random_init=True, task_type='reach')
env.reset_model()

#print(ML1.available_tasks())  # Check out the available tasks

#env = ML1.get_train_tasks('assembly-v1')  # Create an environment with task `pick_place`
#tasks = env.sample_tasks(10)  # Sample a task (in this case, a goal variation)
#print(tasks)
#env.set_task(tasks[1])  # Set task

obs = env.reset()  # Reset environment
#print(obs)
a = env.action_space.sample()
#print (env.get_maxReachDist())
#print(a)
for i in range(5):
    obs = env.reset()
    #print(obs)
    for step in range(10):
        #env.render()
        a = env.action_space.sample()  # Sample an action
        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    print(obs[:3])
    print(env.get_fingerCOM())
    #print(reward)
