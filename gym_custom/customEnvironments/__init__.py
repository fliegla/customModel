from gym.envs.registration import register

register(
  id='DiffDrive-v0',
  entry_point='gym_basic.envs:DiffDriveEnv',
  max_episode_steps=1000,
)
