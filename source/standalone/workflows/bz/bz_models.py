from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import textwrap
import gym
import gymnasium

# pylint: disable-next=unused-import
import flax.linen as nn
import jax

# pylint: disable-next=unused-import
import jax.numpy as jnp

# pylint: disable-next=unused-import
from skrl.models.jax import GaussianMixin

# pylint: disable-next=unused-import
from skrl.models.jax import DeterministicMixin
from skrl.models.jax import Model
from skrl.utils.model_instantiators.jax.common import _parse_input


def create_encoder_string(num):
    return "\n".join(
        [
            "nn.Sequential([",
            f"\tnn.Dense(features={num}),",
            "\tnn.elu,",
            f"\tnn.Dense(features={num}),",
            "\tnn.elu,",
            "])",
        ]
    )


def create_common_agent(inputs):
    encoder_layer_size = 64
    lstm_encoder_size = 128

    networks: list[str] = []
    forward: list[str] = []

    for k, v in inputs.items():
        container_key = f"self.{k}_container"
        networks.append(
            f"{container_key} = {create_encoder_string(encoder_layer_size)}"
        )

        forward.append(f"{k} = {container_key}({_parse_input(v)})")

    networks.extend(
        [
            "",
            f"self.lstm = nn.OptimizedLSTMCell(features={lstm_encoder_size})",
        ]
    )

    forward.extend(
        [
            "",
            "internal_embeddings = jnp.concatenate([priori, context], axis=-1)",
            "final_state_embeddings = internal_embeddings",
            # "lstm_output = internal_embeddings",
            "",
            # pylint: disable-next=line-too-long
            "carry = self.lstm.initialize_carry(jax.random.PRNGKey(0), self.observation_space.shape)",
            "new_carry, lstm_output = self.lstm(carry, final_state_embeddings)",
        ]
    )

    # build substitutions and indent content
    network_string = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    forward_string = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

    return network_string, forward_string


# From the gaussian model
# pylint: disable-next=dangerous-default-value, keyword-arg-before-vararg
def custom_policy_model(
    # pylint: disable-next=unused-argument
    observation_space: Optional[
        Union[int, Tuple[int], gym.Space, gymnasium.Space]
    ] = None,
    action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
    device: Optional[Union[str, jax.Device]] = None,
    clip_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    initial_log_std: float = 0,
    network: Sequence[Mapping[str, Any]] = [],
    # pylint: disable-next=unused-argument
    output: Union[str, Sequence[str]] = "",
    print_source: bool = False,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """Instantiate a Gaussian model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property
                              will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space,
                             gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain
                         the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or jax.Device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param clip_log_std: Flag to indicate whether the log standard deviations
                         should be clipped (default: True)
    :type clip_log_std: bool, optional
    :param min_log_std: Minimum value of the log standard deviation (default: -20)
    :type min_log_std: float, optional
    :param max_log_std: Maximum value of the log standard deviation (default: 2)
    :type max_log_std: float, optional
    :param initial_log_std: Initial value for the log standard deviation (default: 0)
    :type initial_log_std: float, optional
    :param network: Network definition (default: [])
    :type network: list of dict, optional
    :param output: Output expression (default: "")
    :type output: list or str, optional
    :param print_source: Whether to print the source string containing the model class used to
                          instantiate the model (default: False).
    :type print_source: bool, optional

    :return: Gaussian model instance or definition source
    :rtype: Model
    """

    network_string, forward_string = create_common_agent(inputs=network)

    # output_size = output["size"]
    output_size = "self.num_actions"

    template = f"""class CustomPolicyModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

    def setup(self):
        {network_string}

        self.output_layer = nn.Dense(features={output_size})

        self.log_std_parameter = self.param("log_std_parameter", lambda _: {initial_log_std} * jnp.ones({output_size}))

    def __call__(self, inputs, role):
        {forward_string}

        output = self.output_layer(lstm_output)

        return output, self.log_std_parameter, {{}}
    """
    # return source
    if print_source:
        print("--------------------------------------------------\n")
        print(template)
        print("--------------------------------------------------")

    # instantiate model
    _locals = {}
    # pylint: disable-next=exec-used
    exec(template, globals(), _locals)
    return _locals["CustomPolicyModel"](
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
    )


# From the deterministic model
# pylint: disable-next=dangerous-default-value, keyword-arg-before-vararg
def custom_value_model(
    # pylint: disable-next=unused-argument
    observation_space: Optional[
        Union[int, Tuple[int], gym.Space, gymnasium.Space]
    ] = None,
    action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
    device: Optional[Union[str, jax.Device]] = None,
    clip_actions: bool = False,
    network: Sequence[Mapping[str, Any]] = [],
    # pylint: disable-next=unused-argument
    output: Union[str, Sequence[str]] = "",
    print_source: bool = False,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """Instantiate a deterministic model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property
                              will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space,
                             gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the
                         size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or jax.Device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param network: Network definition (default: [])
    :type network: list of dict, optional
    :param output: Output expression (default: "")
    :type output: list or str, optional
    :param print_source: Whether to return the source string containing the model class used to
                          instantiate the model (default: False).
    :type print_source: bool, optional

    :return: Deterministic model instance or definition source
    :rtype: Model
    """

    network_string, forward_string = create_common_agent(inputs=network)

    # output_size = output["size"]
    output_size = 1

    template = f"""class CustomValueModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    def setup(self):
        {network_string}

        self.output_layer = nn.Dense(features={output_size})

    def __call__(self, inputs, role):
        {forward_string}

        output = self.output_layer(lstm_output)

        return output, {{}}
    """
    # return source
    if print_source:
        print("==================================================")
        print(template)
        print("==================================================")

    # instantiate model
    _locals = {}
    # pylint: disable-next=exec-used
    exec(template, globals(), _locals)
    return _locals["CustomValueModel"](
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
    )
