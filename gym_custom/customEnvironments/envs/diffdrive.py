import gym

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

import numpy as np

"""DEFAULT_CAMERA_CONFIG = {}"""

class DiffDriveEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 30,
    }
    
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        MujocoEnv.__init__(self, "/content/customModel/gym_custom/customEnvironments/envs/diffdrive.xml", 5, observation_space=observation_space, **kwargs)

    def step(self, action):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(action, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        return (obs, reward, False, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl))

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:2],
                self.data.qpos.flat[8:10],
                self.data.qvel.flat[:2],
                self.get_body_com("chassis") - self.get_body_com("goal"),
            ]
        )
