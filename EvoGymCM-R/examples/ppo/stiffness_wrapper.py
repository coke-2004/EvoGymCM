import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StiffnessActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, body: np.ndarray, s_low: float = 0.5, s_high: float = 2.0):
        super().__init__(env)
        self._num_actuators = int(np.prod(env.action_space.shape))
        self._num_voxels = int(body.size)

        act_low = np.full(self._num_actuators, 0.6, dtype=float)
        act_high = np.full(self._num_actuators, 1.6, dtype=float)
        s_low_vec = np.full(self._num_voxels, s_low, dtype=float)
        s_high_vec = np.full(self._num_voxels, s_high, dtype=float)

        self.action_space = spaces.Box(
            low=np.concatenate([act_low, s_low_vec], axis=0),
            high=np.concatenate([act_high, s_high_vec], axis=0),
            dtype=float,
        )

    def action(self, act: np.ndarray):
        flat = np.asarray(act, dtype=float).flatten()
        action = flat[: self._num_actuators]
        s_voxel = flat[self._num_actuators :]
        return (action, s_voxel)