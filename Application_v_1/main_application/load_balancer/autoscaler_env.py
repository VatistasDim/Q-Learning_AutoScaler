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
from discretizer import Discretizer
from costs import Costs

service_name = 'mystack_application'
cooldownTimeInSec = 30
w_adp = 0.3
w_perf = 0.5
w_res = 0.4
c_res = 0.5
K_max = 10
Rmax = 1
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
                response_time = int(float(metrics[3]))
                return cpu_percent, ram_percent, time_up, response_time
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

def Calculate_total_cost(w_adp, w_perf, w_res, max_replicas, action, num_containers, cpu_shares, response_time):
    adaptation_cost = Costs.calculate_adaptation_cost(w_adp, action=action)
    performance_penalty = Costs.calculate_performance_penalty(response_time, w_perf, Rmax)
    resource_cost = Costs.calculate_resource_cost(w_res, num_containers, cpu_shares, max_replicas, c_res)
    print(f'w_adp={w_adp}, adaptation_cost={adaptation_cost}, w_perf={w_perf}, perfomance_penaly={performance_penalty}, w_res{w_res}, resource_cost:{resource_cost}')
    total_cost = w_adp * adaptation_cost + w_perf * performance_penalty + w_res * resource_cost
    if total_cost != 0:
        total_cost = round(total_cost, 4)
    return total_cost

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
    def __init__(self, service_name, min_replicas, max_replicas, num_states, max_time_minutes=10):
        super(AutoscaleEnv, self).__init__()

        self.service_name = service_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.num_states = num_states
        self.max_time_minutes = max_time_minutes
        self.start_time = time.time()

        self.action_space = spaces.Discrete(3)  # 3 actions: 0 (scale_out) and 1 (scale_in) and 2 (do nothing)
        #self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)  # (CPU, RAM)

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
            # new_cpu_shares_after_action = self.docker_api.get_stack_containers_mean_cpu_shares()
            # print(f'cpu_shares:{new_cpu_shares_after_action}')
            
        elif action == 1:  # scale_in
            scale_in_action(service_name=self.service_name, min_replicas=self.min_replicas)
            # new_cpu_shares_after_action = self.docker_api.get_stack_containers_mean_cpu_shares()
            # print(f'cpu_shares:{new_cpu_shares_after_action}')
            
        elif action == 2: # do_nothing
            #new_cpu_shares_after_action = self.docker_api.get_stack_containers_mean_cpu_shares()
            print(f'Do nothing')
        
        while True:
            tuple_data = fetch_data()
            has_data = all(ele is None for ele in tuple_data)
            if not has_data:
                cpu_shares, _, _, response_time= tuple_data
                print(f'response_time:{response_time}')
                number_of_containers = get_current_replica_count(service_name)
                discretized_num_containers = Discretizer.discretize_num_containers(number_of_containers, 10, num_states=self.num_states)
                discretized_cpu_share = Discretizer.discretize_stack_containers_cpu_shares(cpu_shares, 100, self.num_states)
                reward = Calculate_total_cost(w_adp=w_adp,
                                              w_perf=w_perf,
                                              w_res=w_res,
                                              max_replicas=10,
                                              action=(action,),
                                              num_containers=discretized_num_containers,
                                              cpu_shares=discretized_cpu_share,
                                              response_time=response_time)
                current_state = self._get_observation()
                return current_state, reward, self._is_done(), {}
            time.sleep(cooldownTimeInSec)
            
    def _get_observation(self):
            max_num_containers = 10
            max_cpu_shares = 100
            while True:
                tuple_data = fetch_data()
                if tuple_data is not None and any(ele is not None for ele in tuple_data):
                    cpu_value, _, _ ,_= tuple_data
                    num_containers = get_current_replica_count(self.service_name)
                    discretized_num_containers = Discretizer.discretize_num_containers(num_containers, max_num_containers, num_states=self.num_states)
                    discretized_cpu_share = Discretizer.discretize_stack_containers_cpu_shares(cpu_value, max_cpu_shares, self.num_states)
                    return discretized_num_containers, float(cpu_value), discretized_cpu_share
                time.sleep(cooldownTimeInSec)


    def _is_done(self):
        """
        Check if the episode is done based on elapsed time.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        elapsed_time_minutes = (time.time() - self.start_time) / 60.0
        return elapsed_time_minutes >= self.max_time_minutes
