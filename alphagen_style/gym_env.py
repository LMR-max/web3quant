from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from alphagen_style.evaluation import EvalConfig, RewardConfig
from alphagen_style.masking import ActionSpace
from alphagen_style.rl_env import EnvConfig, MaskedAlphaEnv


@dataclass(frozen=True)
class GymObsConfig:
    max_steps: int = 64


class GymMaskedAlphaEnv(gym.Env):
    """Gymnasium wrapper around MaskedAlphaEnv compatible with sb3-contrib ActionMasker/MaskablePPO."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        panel: pd.DataFrame,
        action_space: ActionSpace,
        targets: Sequence[str],
        env_cfg: EnvConfig = EnvConfig(),
        eval_cfg: Optional[EvalConfig] = None,
        reward_cfg: RewardConfig = RewardConfig(),
        obs_cfg: GymObsConfig = GymObsConfig(),
    ):
        super().__init__()
        self._core = MaskedAlphaEnv(
            panel=panel,
            action_space=action_space,
            targets=targets,
            env_cfg=env_cfg,
            eval_cfg=eval_cfg,
            reward_cfg=reward_cfg,
        )
        self._obs_cfg = obs_cfg

        # Discrete action space
        self.action_space = spaces.Discrete(self._core.action_space.n)

        # Numeric observation only (vector) to keep PPO happy.
        # [stack_len, top_kind_id, top_depth, steps, *action_history_10]
        # 4 basic features + 10 history = 14
        self.observation_space = spaces.Box(low=0.0, high=1e9, shape=(14,), dtype=np.float32)

        self._kind_to_id = {
            "series": 0,
            "scalar": 1,
            "window": 2,
            "terminal": 3,
        }

    def _obs_to_vec(self, obs: Dict[str, object]) -> np.ndarray:
        stack_len = float(obs.get("stack_len", 0))
        top_kind = str(obs.get("top_kind", "terminal"))
        top_kind_id = float(self._kind_to_id.get(top_kind, 3))
        top_depth = float(obs.get("top_depth", 0))
        steps = float(obs.get("steps", 0))
        
        hist = obs.get("history", [])
        if not isinstance(hist, list):
            hist = []
        # Pad history to 10
        pad = [0.0] * (10 - len(hist))
        hist_vec = [float(x) for x in hist] + pad
        
        return np.array([stack_len, top_kind_id, top_depth, steps] + hist_vec, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._core.reset(seed=seed)
        return self._obs_to_vec(obs), info

    def step(self, action: int):
        obs, reward, done, info = self._core.step(int(action))
        terminated = bool(done)
        truncated = bool(obs.get("steps", 0) >= self._obs_cfg.max_steps)
        return self._obs_to_vec(obs), float(reward), terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        # sb3-contrib ActionMasker will call this.
        return self._core.action_masks().astype(bool)
