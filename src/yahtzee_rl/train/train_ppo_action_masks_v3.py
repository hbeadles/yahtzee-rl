from yahtzee_rl.envs.yahtzee_env import YahtzeeEnv
from yahtzee_rl.train.train_baselines import TrainerBaselines, ModelType

env = YahtzeeEnv(lambda_upper=0.05, lambda_yahtzee=0.2, use_expecteds=False, use_probabilities=False)
policy_kwargs = dict(
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_timesteps = 30e6
# path = "experiments/ppo_yahtzee_v3_full_no_expecteds_no_probabilities/2026-03-22/model.zip"
# vecnormalize_path = "experiments/ppo_yahtzee_v3_full_no_expecteds_no_probabilities/2026-03-22/vecnormalize.pkl"
trainer = TrainerBaselines(ModelType.MASKABLE_PPO, 
                            env, 'ppo_yahtzee_v3_full_no_expecteds_no_probabilities', batch_size=128, n_steps=2512,
                            gamma=0.99, n_epochs=8, policy_kwargs=policy_kwargs, ent_coef=0.03,
                            vec_normalize=True, clip_range=0.1, gae_lambda=(0.25, 0.97), normalize_advantage=False)
# trainer.load(model_path=path, vecnormalize_path=vecnormalize_path)
trainer.train(max_timesteps=max_timesteps, save_freq=500000)
mean_reward, std_reward = trainer.evaluate(num_episodes=10)
print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
trainer.plot_results(max_timesteps=max_timesteps) 