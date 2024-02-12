"""
Autoscaling Environment

This script provides an environment for autoscaling a Docker service using OpenAI Gym. The environment includes actions for scaling in and scaling out, and calculates rewards based on CPU and RAM metrics.

Usage:
- Instantiate the AutoscaleEnv class with the required parameters.
- Use the reset() method to initialize the environment.
- Use the step(action) method to take an action (0 for scale_out, 1 for scale_in) and observe the new state, reward, and whether the environment is done.
"""

import gym
import time
from gym import spaces
import numpy as np
import prometheus_metrics
import time, docker

service_name = 'mystack_application'
cooldownTimeInSec = 30

client = docker.from_env()
clients_list = client.services.list()

def get_current_replica_count(service_prefix):
    """
    Gets the replicas number from Docker.

    Args:
        service_prefix (str): The prefix of the service name.

    Returns:
        int or None: The number of replicas if found, otherwise None.
    """
    client = docker.from_env()
    try:
        for service in client.services.list():
            if service_prefix in service.name:
                return service.attrs['Spec']['Mode']['Replicated']['Replicas']
        return None
    except docker.errors.NotFound:
        return None

def scale_in_action(service_name, min_replicas):
    """
    Logic for the scale in action.

    Args:
        service_name (str): Name of the service to scale.
        min_replicas (int): Minimum number of replicas allowed.

    Returns:
        int or None: The new replica count after scaling in, or None if scaling is not possible.
    """
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is not None and current_replicas > min_replicas:
        current_replicas -= 1
        print(f"Horizontal Scale In: Replicas decreased to: {current_replicas}, system waits 5 seconds")
        scale_in(service_name, 1)
        return current_replicas
    else:
        print(f"Already at minimum replicas: {min_replicas}")
        return None

def scale_out_action(service_name, max_replicas):
    """
    Logic for the scale out action.

    Args:
        service_name (str): Name of the service to scale.
        max_replicas (int): Maximum number of replicas allowed.

    Returns:
        int or None: The new replica count after scaling out, or None if scaling is not possible.
    """
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is not None and current_replicas < max_replicas:
        current_replicas = current_replicas + 1
        print(f"Horizontal Scale Out: Replicas increased to: {current_replicas}")  # Print scale out message
        scale_out(service_name, current_replicas)  # Increase replicas
    else:
        print("Already at maximum replicas.")  # Indicate already at maximum replicas
        #reset_replicas(service_name=service_name)

    return current_replicas

def scale_out(service_name, desired_replicas):
    """
    Scales out a service to the specified number of replicas.
    """
    client = docker.from_env()
    service = client.services.get(service_name)
    service.scale(desired_replicas)
    print(f"Service '{service_name}' scaled to {desired_replicas} replicas.")
    time.sleep(cooldownTimeInSec)

def scale_in(service_name, scale_out_factor):
    """
    Scales in a service to the specified number of replicas.

    Args:
        service_name (str): Name of the service to scale.
        scale_out_factor (int): The number of replicas to scale in by.
    """
    client = docker.from_env()
    service = client.services.get(service_name)
    current_replicas = get_current_replica_count(service_name)
    desired_replicas = current_replicas - scale_out_factor
    service.scale(desired_replicas)
    time.sleep(cooldownTimeInSec)

def get_reward(cpu_value, ram_value, cpu_threshold, ram_threshold):
    """
    Calculates the reward based on CPU and RAM metrics.

    Args:
        cpu_value (int): Current CPU utilization percentage.
        ram_value (int): Current RAM utilization percentage.
        cpu_threshold (int): CPU threshold for autoscaling.
        ram_threshold (int): RAM threshold for autoscaling.

    Returns:
        int: The calculated reward.
    """
    if cpu_value is not None and ram_value is not None:
        are_too_many_containers = False
        close_to_achieve_reward = False
        cpu_threshold_20_percent = cpu_value * 0.20
        ram_threshold_20_percent = ram_value * 0.20
        cpu_threshold_merged = cpu_threshold + int(cpu_threshold_20_percent)
        ram_threshold_merged = ram_threshold + int(ram_threshold_20_percent)
        print(f"CPU_Plus_20%: {cpu_threshold_merged} RAM_Plus_20%: {ram_threshold_merged}")

        cpu_diff = cpu_threshold_merged - cpu_value
        cpu_diff_low = cpu_value - cpu_threshold_merged
        if cpu_diff >= 10:
            are_too_many_containers = True
        elif cpu_diff_low <= 15:
            close_to_achieve_reward = True
        
        if cpu_value <= cpu_threshold_merged and ram_value <= ram_threshold_merged:
            print(f"Reward={20}, cpu_value={cpu_value} <= {cpu_threshold_merged} and ram_value={ram_value} <= {ram_threshold_merged}")
            return 20
        elif close_to_achieve_reward:
            if are_too_many_containers:
                print(f"Reward {-10}: Caused by, Too many containers are running.")
                return -10
            print("Close to achieve reward!")
            print(f"Reward{-5}, cpu_value={cpu_value} <= {cpu_threshold_merged} and ram_value={ram_value} <= {ram_threshold_merged}")
            return -5
        elif are_too_many_containers:
            print(f"Reward {-10}: Caused by, Too many containers are running.")
            return -10
        else:
            print(f"Reward={-15}, cpu_value={cpu_value} >= {cpu_threshold_merged} or ram_value={ram_value} >= {ram_threshold_merged}")
            return -15
    else:
        print(f"Reward={0}: Caused by, There was an error when trying to calculate the reward function")
        return 0

def fetch_data():
    """
    Fetches data from the Prometheus metrics API.

    Returns:
        tuple: A tuple containing CPU percent, RAM percent, and uptime, or None if there is an error.
    """
    try:
        metrics = prometheus_metrics.start_metrics_service(url=url)
        if metrics is not None:
            time_up = metrics[2]
            if time_up != '0':
                cpu_percent = int(float(metrics[0]))
                ram_percent = int(float(metrics[1]))
                time_up = int(float(time_up))
                return cpu_percent, ram_percent, time_up
        return None, None, None
    except Exception as e:
        print("An error occurred during service metrics retrieval:", e)
        return None, None, None 

def reset_replicas(service_name):
    """
    Resets the service to one replica.

    Args:
        service_name (str): Name of the service to reset.
    """
    client = docker.from_env()
    service = client.services.get(service_name)
    desired_replicas = 1
    service.scale(desired_replicas)
    time.sleep(cooldownTimeInSec)

def Calculate_Thresholds():
    """
    Calculates the CPU and RAM thresholds based on the current number of replicas.

    Returns:
        tuple: A tuple containing the CPU and RAM thresholds.
    """
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is not None:
        cpu_threshold = 1 + (current_replicas - 1) * 8 if current_replicas <= 10 else 100
        ram_threshold = 10 + (current_replicas - 1) * 8 if current_replicas <= 10 else 100
    else:
        cpu_threshold = 0  # Default value if replicas count is not available
        ram_threshold = 10  # Default value if replicas count is not available

    print(f"Thresholds calculated as CPU:{cpu_threshold}, RAM: {ram_threshold}")
    return cpu_threshold, ram_threshold

url = 'http://prometheus:9090/api/v1/query'

class AutoscaleEnv(gym.Env):
    """
    Autoscaling Environment class for OpenAI Gym.

    Attributes:
        service_name (str): Name of the Docker service.
        min_replicas (int): Minimum number of replicas allowed.
        max_replicas (int): Maximum number of replicas allowed.
        cpu_threshold (int): CPU threshold for autoscaling.
        ram_threshold (int): RAM threshold for autoscaling.
        num_states (int): Number of states in the observation space.
        max_time_minutes (int): Maximum time in minutes for the episode.
        start_time (float): Start time of the episode.
        action_space (gym.Space): Action space for the environment.
        observation_space (gym.Space): Observation space for the environment.
    """
    def __init__(self, service_name, min_replicas, max_replicas, cpu_threshold, ram_threshold, num_states, max_time_minutes=10):
        super(AutoscaleEnv, self).__init__()

        self.service_name = service_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold
        self.num_states = num_states
        self.max_time_minutes = max_time_minutes
        self.start_time = time.time()

        self.action_space = spaces.Discrete(2)  # 2 actions: 0 (scale_out) and 1 (scale_in)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)  # (CPU, RAM)

    def reset(self):
        """
        Reset environment to initial state.

        Returns:
            np.array: Initial observation.
        """
        reset_replicas(service_name=self.service_name)  # Reset Env.
        return self._get_observation()

    def step(self, action):
        """
        Take action and observe new state and reward.

        Args:
            action (int): Action to take (0 for scale_out, 1 for scale_in).

        Returns:
            tuple: Tuple containing the new state, reward, whether the episode is done, and additional info.
        """
        if action == 0:  # scale_out
            scale_out_action(service_name=self.service_name, max_replicas=self.max_replicas)
            self.cpu_threshold, self.ram_threshold = Calculate_Thresholds()
            
        elif action == 1:  # scale_in
            scale_in_action(service_name=self.service_name, min_replicas=self.min_replicas)
            self.cpu_threshold, self.ram_threshold = Calculate_Thresholds()
        
        while True:
            tuple_data = fetch_data()
            has_data = all(ele is None for ele in tuple_data)
            if not has_data:
                cpu_value, ram_value, _ = tuple_data
                reward = get_reward(cpu_value, ram_value, self.cpu_threshold, self.ram_threshold)

                next_state = self._get_observation()

                return next_state, reward, self._is_done(), {}
            time.sleep(cooldownTimeInSec)

    def _get_observation(self):
        """
        Return the current CPU and RAM values as the observation.

        Returns:
            np.array: Current observation.
        """
        while True:
            tuple_data = fetch_data()
            # Check if data is not None and not empty
            if tuple_data is not None and any(ele is not None for ele in tuple_data):
                cpu_value, ram_value, _ = tuple_data  # Implement your metric fetching logic here
                return np.array([cpu_value, ram_value], dtype=np.float32)
            
            # If data is None or empty, wait for a short duration before trying again
            time.sleep(cooldownTimeInSec)  # Adjust the duration based on your requirements

    def _is_done(self):
        """
        Check if the episode is done based on elapsed time.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        elapsed_time_minutes = (time.time() - self.start_time) / 60.0
        return elapsed_time_minutes >= self.max_time_minutes
