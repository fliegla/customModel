from gym.envs.registration import register

register(
  id='DiffDrive-v0',
  entry_point='customEnvironments.envs:DiffDriveEnv',
  max_episode_steps=1000,
)
