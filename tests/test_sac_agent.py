import numpy as np
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sac_agent import Actor, SAC
from src.replay_buffer import ReplayBuffer


@pytest.fixture
def sac_config():
    return {
        'rl_hyperparameters': {
            'gamma': {'default': 0.99},
            'tau': {'default': 0.005},
            'alpha': {'default': 0.2},
            'max_action': 1.0,
            'state_dim': 3,
            'action_dim': 2,
            'network_architecture': {'actor': [16, 16], 'critic': [16, 16]},
            'num_atoms': 5,
            'v_min': -1.0,
            'v_max': 1.0,
            'learning_rate': 1e-3,
        }
    }


def test_actor_sample_shapes():
    actor = Actor(3, 2, 1.0, [16, 16])
    state = torch.zeros(4, 3)
    action, log_prob = actor.sample(state)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4, 1)


def test_sac_training_step(sac_config):
    agent = SAC(sac_config)
    buffer = ReplayBuffer(max_size=10)
    for _ in range(3):
        s = np.zeros(3)
        ns = np.ones(3)
        a = np.zeros(2)
        buffer.add(s, ns, a, 0.0, False)
    agent.train(buffer, batch_size=2)
    assert agent.total_it == 1

