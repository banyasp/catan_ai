"""Tree-search main-play policy (Blue) for evaluation — no PPO / no torch model."""

from __future__ import annotations

import numpy as np
from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index
from catanatron.models.enums import Action
from catanatron.models.player import Color
from catanatron.players.minimax import AlphaBetaPlayer


class AlphaBetaMainPlayAgent:
    """Same select_action/store/train surface as MainPlayAgent; uses AlphaBeta search."""

    def __init__(self, depth: int = 2, prunning: bool = False):
        self._depth = max(1, int(depth))
        self._prunning = bool(prunning)
        self._player = AlphaBetaPlayer(
            Color.BLUE, depth=self._depth, prunning=self._prunning
        )

    def select_action(
        self,
        state,
        mask,
        game=None,
        playable_actions=None,
        deterministic: bool = False,
        **_kwargs,
    ):
        del state, deterministic
        if game is None or playable_actions is None:
            raise ValueError(
                "AlphaBetaMainPlayAgent requires game= and playable_actions= (from simulate_game)"
            )
        acts = list(playable_actions)
        chosen: Action = self._player.decide(game, acts)
        cap_idx = int(catanatron_action_to_capstone_index(chosen))
        mask_arr = np.asarray(mask, dtype=np.float32)
        if cap_idx < 0 or cap_idx >= len(mask_arr) or mask_arr[cap_idx] <= 0.5:
            valid = np.where(mask_arr > 0.5)[0]
            if len(valid) == 0:
                raise RuntimeError("AlphaBetaMainPlayAgent: empty action mask")
            cap_idx = int(valid[0])
        return (cap_idx, 0.0, 0.0)

    def store(self, *args, **kwargs):
        pass

    def train(self, last_value):
        del last_value
        return {}

    def load(self, path=None):
        pass

    def save(self, path=None):
        pass
