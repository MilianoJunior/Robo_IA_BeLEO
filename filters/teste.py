import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)



num_actions = 3 # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)

state = tf.constant([[0,0]])
action_logits_t, value = model(state)


print('state: ',)