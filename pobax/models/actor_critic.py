from flax import linen as nn
from jax import numpy as jnp
from jax._src.nn.initializers import orthogonal, constant

from pobax.models.discrete import DiscreteActor
from pobax.models.continuous import ContinuousActor
from pobax.models.value import Critic
from pobax.models.network import SimpleNN, ScannedRNN, ScannedURNN
from pobax.models.embedding import CNN, BattleshipEmbedding


class ActorCritic(nn.Module):
    env_name: str
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False
    memoryless: bool = False
    is_discrete: bool = True
    is_image: bool = False
    memory_type: str = 'gru'
    urnn_variant: str = 'standard'
    urnn_input_dense: bool = True
    urnn_norm_scale: float = 1.0
    urnn_perm_seed: int = 0

    def setup(self):
        if self.is_image:
            self.embedding = CNN(hidden_size=self.hidden_size)
        # elif 'battleship' in self.env_name:
        #     self.embedding = BattleshipEmbedding(hidden_size=self.hidden_size, action_dim=self.action_dim)
        elif not self.memoryless:
            self.embedding = nn.Sequential([
                nn.Dense(self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
                nn.relu
            ])
        else:
            self.embedding = SimpleNN(hidden_size=self.hidden_size)

        if not self.memoryless:
            if self.memory_type == 'urnn':
                self.memory = ScannedURNN(
                    hidden_size=self.hidden_size,
                    variant=self.urnn_variant,
                    add_input_dense=self.urnn_input_dense,
                    norm_scale=self.urnn_norm_scale,
                    perm_seed=self.urnn_perm_seed,
                )
            else:
                self.memory = ScannedRNN(hidden_size=self.hidden_size)
        if self.is_discrete:
            self.actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        else:
            self.actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size)

        if self.double_critic:
            self.critic = nn.vmap(Critic,
                                  variable_axes={'params': 0},
                                  split_rngs={'params': True},
                                  in_axes=None,
                                  out_axes=2,
                                  axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.critic = Critic(hidden_size=self.hidden_size)

    def __call__(self, hidden, x):
        obs_dict, dones = x
        obs = obs_dict.obs
        action_mask = obs_dict.action_mask
        embedding = self.embedding(obs)
        if not self.memoryless:
            rnn_in = (embedding, dones)
            hidden, mem_out = self.memory(hidden, rnn_in)
            # QuRNN hook: a future --policy_head born flag will replace this
            # real-concat adapter with a complex actor head.
            if self.memory_type == 'urnn':
                embedding = jnp.concatenate([mem_out.real, mem_out.imag], axis=-1)
            else:
                embedding = mem_out

        pi = self.actor(embedding, action_mask=action_mask)
        v = self.critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)
