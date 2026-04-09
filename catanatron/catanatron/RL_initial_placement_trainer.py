"""Hybrid games: CapstoneModel vs a placement opponent during initial setup, then new bots for the rest."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from catanatron.game import TURNS_LIMIT, Game
from catanatron.gym.envs.capstone_env import (
    ACTION_SPACE_SIZE,
    CapstoneCatanatronEnv,
    simple_reward,
)
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_capstone_agent = _REPO_ROOT / "capstone_agent"
if _capstone_agent.is_dir():
    _cap_path = str(_capstone_agent)
    if _cap_path not in sys.path:
        sys.path.insert(0, _cap_path)

from CapstoneModel import CapstoneModel  # noqa: E402

OBS_SIZE = 1258
DEFAULT_HIDDEN = 512
DEFAULT_ALPHABETA_DEPTH = 2
DEFAULT_ALPHABETA_PRUNING = True


def valid_actions_to_mask(valid_actions: Sequence[int], size: int = ACTION_SPACE_SIZE) -> np.ndarray:
    mask = np.zeros(size, dtype=np.float32)
    for a in valid_actions:
        if 0 <= a < size:
            mask[a] = 1.0
    return mask


def make_capstone_env(
    enemies: Optional[List] = None,
    **config_overrides,
) -> CapstoneCatanatronEnv:
    cfg = dict(enemies=enemies if enemies is not None else [RandomPlayer(Color.RED)])
    cfg.update(config_overrides)
    return CapstoneCatanatronEnv(config=cfg)


def sample_model_action(
    model: CapstoneModel,
    observation: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> int:
    with torch.no_grad():
        x = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
        probs, _ = model(x, m)
    dist = Categorical(probs)
    return int(dist.sample().item())


def default_post_phase_players(
    game: Game,
    depth: int = DEFAULT_ALPHABETA_DEPTH,
    prunning: bool = DEFAULT_ALPHABETA_PRUNING,
) -> List[AlphaBetaPlayer]:
    """One AlphaBeta per seat (same order as ``game.state.colors``)."""
    return [
        AlphaBetaPlayer(color, depth, prunning)
        for color in game.state.colors
    ]


def install_players(game: Game, players: Sequence[Player]) -> None:
    """Replace ``game.state.players``; each ``players[i].color`` must match ``game.state.colors[i]``."""
    if len(players) != len(game.state.players):
        raise ValueError(
            f"post_phase_players length {len(players)} != num seats {len(game.state.players)}"
        )
    for i, p in enumerate(players):
        if p.color != game.state.colors[i]:
            raise ValueError(
                f"post_phase_players[{i}] has color {p.color}, expected {game.state.colors[i]}"
            )
    game.state.players = list(players)


def run_hybrid_episode(
    model: CapstoneModel,
    env: CapstoneCatanatronEnv,
    device: torch.device,
    post_phase_players: Optional[Sequence[Player]] = None,
    seed: Optional[int] = None,
) -> Tuple[float, int, bool]:
    """
    Initial build phase (settlements + roads): CapstoneModel for BLUE via ``env.step``;
    placement opponent(s) act via env ``play_tick``. After ``is_initial_build_phase`` ends,
    swap in ``post_phase_players`` (default: AlphaBeta vs AlphaBeta) and finish with ``play_tick``.

    Returns (cumulative_reward, total_decision_steps, won_for_blue).
    """
    obs, info = env.reset(seed=seed)
    mask = valid_actions_to_mask(info["valid_actions"])
    total_reward = 0.0
    steps = 0
    done = False

    # --- Phase 1: initial settlement + roads (model vs placement opponent) ---
    while env.game.state.is_initial_build_phase and not done:
        action = sample_model_action(model, obs, mask, device)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
        mask = valid_actions_to_mask(info["valid_actions"])

    if done:
        won = env.game.winning_color() == env.self_player.color
        return total_reward, steps, won

    # --- Phase 2: replace all seats, then bot-vs-bot to completion ---
    game = env.game
    if post_phase_players is None:
        new_players = default_post_phase_players(game)
    else:
        new_players = list(post_phase_players)
    install_players(game, new_players)

    while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
        game.play_tick()
        steps += 1

    final_r = float(simple_reward(game, env.self_player.color))
    total_reward += final_r
    won = game.winning_color() == env.self_player.color
    return total_reward, steps, won


def play_against_opponent(
    num_games: int,
    model: Optional[CapstoneModel] = None,
    placement_opponent: Optional[Player] = None,
    post_phase_players: Optional[Sequence[Player]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> None:
    """
    ``placement_opponent``: bot BLUE faces during initial setup (default ``RandomPlayer(Color.RED)``).

    ``post_phase_players``: one ``Player`` per seat after setup, in ``game.state.colors`` order
    (default: ``AlphaBetaPlayer`` for each color).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = CapstoneModel(OBS_SIZE, DEFAULT_HIDDEN)
    model = model.to(device)
    model.eval()

    enemies = [placement_opponent] if placement_opponent is not None else None
    env = make_capstone_env(enemies=enemies)

    wins = 0
    for g in range(num_games):
        ep_seed = (seed + g) if seed is not None else None
        total_reward, steps, won = run_hybrid_episode(
            model,
            env,
            device,
            post_phase_players=post_phase_players,
            seed=ep_seed,
        )
        wins += int(won)
        print(
            f"game {g + 1}/{num_games}  steps={steps}  reward={total_reward:.3f}  win={won}"
        )

    print(f"summary: {wins}/{num_games} wins")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="CapstoneModel vs placement opponent, then AlphaBeta vs AlphaBeta"
    )
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    play_against_opponent(num_games=args.games, seed=args.seed)
