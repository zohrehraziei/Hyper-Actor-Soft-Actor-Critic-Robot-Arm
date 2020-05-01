# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:13:42 2020

@author: Zohreh Raziei
"""

#from rllab.algos.trpo import TRPO
#from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from rllab.envs.box2d.cartpole_env import CartpoleEnv
#from rllab.envs.normalized_env import normalize
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
#
#env = normalize(CartpoleEnv())
#
#policy = GaussianMLPPolicy(
#    env_spec=env.spec,
#    # The neural network policy should have two hidden layers, each with 32 hidden units.
#    hidden_sizes=(32, 32)
#)
#
#baseline = LinearFeatureBaseline(env_spec=env.spec)
#
#algo = TRPO(
#    env=env,
#    policy=policy,
#    baseline=baseline,
#    batch_size=4000,
#    whole_paths=True,
#    max_path_length=100,
#    n_itr=40,
#    discount=0.99,
#    step_size=0.01,
#)
#algo.train()

from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalMLPPolicy


def run_task(snapshot_config, *_):
    """Wrap TRPO training task in the run_task function."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode="last",
    seed=1,
)