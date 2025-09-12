import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.replay_buffer import ReplayBuffer


def test_add_and_sample():
    """ReplayBuffer stores experiences and samples them correctly."""
    buffer = ReplayBuffer(max_size=2)
    s1 = np.array([1.0, 2.0])
    buffer.add(s1, s1, np.array([0.1]), 1.0, False)
    s2 = np.array([3.0, 4.0])
    buffer.add(s2, s2, np.array([0.2]), 2.0, True)

    assert len(buffer.storage) == 2
    state, next_state, action, reward, done = buffer.sample(2)
    assert state.shape == (2, 2)
    assert action.shape == (2, 1)
    assert reward.shape == (2,)
    assert done.shape == (2,)


def test_sample_undersized_buffer():
    """Sampling more elements than present should raise ValueError."""
    buffer = ReplayBuffer(max_size=1)
    s = np.array([0.0])
    buffer.add(s, s, s, 0.0, False)
    with pytest.raises(ValueError):
        buffer.sample(2)


def test_sample_empty_buffer():
    """Sampling from an empty buffer should raise ValueError."""
    buffer = ReplayBuffer()
    with pytest.raises(ValueError):
        buffer.sample(1)

