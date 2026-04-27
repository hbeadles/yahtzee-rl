from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.train.train_baselines import TrainerBaselines, ModelType

env = YahtzeeEnv(lambda_upper=0.05, lambda_yahtzee=0.2, use_expecteds=False,
  invalid_action_substitute=True, invalid_action_penalty=-20.0)
policy_kwargs = dict(
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_timesteps = 30e6
trainer = TrainerBaselines(ModelType.A2C, 
                            env, 'a2c_yahtzee_v3_full', n_steps=2512,
                            gamma=0.99, policy_kwargs=policy_kwargs, ent_coef=0.02,
                            vec_normalize=True, gae_lambda=(0.3, 0.95), normalize_advantage=False)

trainer.train(max_timesteps=max_timesteps, save_freq=100000)
mean_reward, std_reward = trainer.evaluate(num_episodes=10)
print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
trainer.plot_results(max_timesteps=max_timesteps) 