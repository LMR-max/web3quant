from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alphagen_style.dsl import analyze_expr, eval_expr
from alphagen_style.evaluation import EvalConfig, RewardConfig, compute_reward, evaluate_factor_panel
from alphagen_style.masking import Action, ActionSpace, ExprState, initial_state, actions_to_expr


@dataclass(frozen=True)
class EnvConfig:
    max_depth: int = 3
    n_folds: int = 5
    embargo_bars: int = 60


class MaskedAlphaEnv:
    """Minimal RL environment skeleton (no gym dependency).

    - State: pending slots stack (ExprState)
    - Action: integer index into ActionSpace.actions
    - Done: when stack is empty (expression complete)
    - Reward: computed at terminal by evaluate_factor_panel + compute_reward

    This is designed to be wrapped later by gymnasium + sb3-contrib MaskablePPO.
    """

    def __init__(
        self,
        *,
        panel: pd.DataFrame,
        action_space: ActionSpace,
        targets: Sequence[str],
        env_cfg: EnvConfig = EnvConfig(),
        eval_cfg: Optional[EvalConfig] = None,
        reward_cfg: RewardConfig = RewardConfig(),
        factor_name: str = "alpha",
        cache_size: int = 2048,
    ):
        self.panel = panel
        self.action_space = action_space
        self.targets = list(targets)
        self.env_cfg = env_cfg
        self.eval_cfg = eval_cfg or EvalConfig()
        self.reward_cfg = reward_cfg
        self.factor_name = factor_name

        self._state: ExprState = initial_state(max_depth=self.env_cfg.max_depth)
        self._actions: List[Action] = []
        self._rng = np.random.default_rng(7)

        self._cache_size = int(cache_size)
        self._reward_cache: Dict[str, float] = {}
        self._cache_fifo: List[str] = []

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Dict[str, object], Dict[str, object]]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._state = initial_state(max_depth=self.env_cfg.max_depth)
        self._actions = []
        # History of action indices (padded with -1)
        self._action_history: List[int] = []
        
        obs = self._get_obs()
        info = {"action_mask": self.action_masks()}
        return obs, info

    def _get_obs(self) -> Dict[str, object]:
        # Keep it simple: expose stack size and remaining depth of top slot.
        if self._state.stack:
            slot, depth = self._state.stack[-1]
            top_kind = slot.kind
            top_depth = depth
        else:
            top_kind = "terminal"
            top_depth = 0
            
        return {
            "stack_len": int(len(self._state.stack)),
            "top_kind": top_kind,
            "top_depth": int(top_depth),
            "steps": int(len(self._actions)),
            "history": self._action_history[-10:] if self._action_history else [],  # Last 10 actions
        }

    def action_masks(self) -> np.ndarray:
        return self.action_space.mask(self._state)

    def step(self, action_index: int) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        a = self.action_space.actions[int(action_index)]
        self._actions.append(a)
        self._action_history.append(int(action_index))

        from alphagen_style.masking import step as _step

        _step(self._state, a, max_depth=self.env_cfg.max_depth)

        done = len(self._state.stack) == 0
        if not done:
            return self._get_obs(), 0.0, False, {"action_mask": self.action_masks()}

        expr = actions_to_expr(self.action_space, self._actions, max_depth=self.env_cfg.max_depth)
        reward = self._score_expr(expr)
        info = {
            "expr": expr,
            "n_actions": int(len(self._actions)),
            "action_mask": self.action_masks(),
        }
        return self._get_obs(), float(reward), True, info

    def _score_expr(self, expr: str) -> float:
        # small cache
        if expr in self._reward_cache:
            return self._reward_cache[expr]

        try:
            info = analyze_expr(expr)
            df = self.panel.copy()
            df[self.factor_name] = eval_expr(expr, df)

            # Drop warmup bars so rolling NaNs don't dominate missingness/metrics.
            df_eval = df.iloc[int(info.warmup_bars) :].copy() if int(info.warmup_bars) > 0 else df

            res = evaluate_factor_panel(
                df_eval,
                factor_col=self.factor_name,
                target_cols=self.targets,
                cfg=self.eval_cfg,
                trade_cfg=self.reward_cfg,
            )
            r = compute_reward(res, cfg=self.reward_cfg)
            reward = float(r["reward"])
        except Exception:
            # Never crash training because an expression is numerically bad.
            reward = -10.0

        # maintain cache
        self._reward_cache[expr] = reward
        self._cache_fifo.append(expr)
        if len(self._cache_fifo) > self._cache_size:
            old = self._cache_fifo.pop(0)
            self._reward_cache.pop(old, None)

        return reward
