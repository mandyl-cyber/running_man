from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name: str) -> type[nn.Module]:
    if name == "elu":
        return nn.ELU
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(in_dim: int, out_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    act = _activation(activation)
    layers: list[nn.Module] = []
    last_dim = in_dim
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(last_dim, hidden_dim), act()]
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class StudentDepthActorCritic(nn.Module):
    """Actor-critic with a 3-layer depth CNN and GAP for the student actor."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        obs_groups: dict[str, list[str]] | None = None,
        depth_height: int = 58,
        depth_width: int = 87,
        depth_dim: int = 32,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 0.01,
    ):
        super().__init__()
        actor_hidden_dims = actor_hidden_dims or [2048, 1024, 512, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]
        self.num_actions = num_actions
        self.obs_groups = obs_groups or {"policy": ["policy"], "critic": ["critic"]}
        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_numel = depth_height * depth_width
        self.depth_start = num_actor_obs - self.depth_numel - num_actions
        self.depth_end = self.depth_start + self.depth_numel
        if self.depth_start <= 0:
            raise ValueError(
                f"Actor obs dim {num_actor_obs} is too small for depth image {depth_height}x{depth_width} "
                f"and {num_actions} previous-action values."
            )

        act = _activation(activation)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            act(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            act(),
            nn.Conv2d(32, depth_dim, kernel_size=3, stride=2, padding=1),
            act(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        proprio_dim = num_actor_obs - self.depth_numel
        self.actor = _mlp(proprio_dim + depth_dim, num_actions, actor_hidden_dims, activation)
        self.critic = _mlp(num_critic_obs, 1, critic_hidden_dims, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None):
        pass

    def _is_obs_mapping(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> bool:
        return not isinstance(obs, torch.Tensor) and all(group in obs for group in self.obs_groups)

    def _get_actor_obs(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if self._is_obs_mapping(obs):
            return torch.cat([obs[name] for name in self.obs_groups["policy"]], dim=-1)
        return obs

    def _get_critic_obs(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if self._is_obs_mapping(obs):
            return torch.cat([obs[name] for name in self.obs_groups["critic"]], dim=-1)
        return obs

    def _actor_input(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self._get_actor_obs(obs)
        depth = obs[:, self.depth_start : self.depth_end].reshape(-1, 1, self.depth_height, self.depth_width)
        depth_features = self.depth_encoder(depth)
        proprio = torch.cat((obs[:, : self.depth_start], obs[:, self.depth_end :]), dim=-1)
        return torch.cat((proprio, depth_features), dim=-1)

    def _update_distribution(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        mean = self.actor(self._actor_input(obs))
        # RSL-RL updates this parameter directly.  Without a lower bound it can
        # cross zero late in training and torch.distributions.Normal then
        # aborts with "normal expects all elements of std >= 0.0".
        # Keep the legacy checkpoint layout while enforcing a valid scale.
        std = torch.clamp(self.std, min=1.0e-6).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(self._actor_input(obs))

    def evaluate(self, obs: torch.Tensor | dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self.critic(self._get_critic_obs(obs))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        pass
