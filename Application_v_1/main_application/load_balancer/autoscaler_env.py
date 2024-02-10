import gym, time
from gym import spaces
import numpy as np
import prometheus_metrics
import time, docker

service_name = 'mystack_application'
service_produce_load_name = 'mystack_produce-load'
cooldownTimeInSec = 30

client = docker.from_env()
clients_list = client.services.list()
def get_current_replica_count(service_prefix):
    """
    Gets the replicas number from Docker.
    Args:
        service_prefix (_type_): _description_

    Returns:
        _type_: _description_
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
        tuple: Tuple containing the updated rewards list and the new replica count.
    """
    #tuple_data = fetch_data() # Fetch new data
    current_replicas = get_current_replica_count(service_name)  # Get current replica count
    if current_replicas is not None and current_replicas > min_replicas:  # If current replicas is not None and greater than min replicas
        current_replicas = current_replicas - 1
        print(f"Horizontal Scale In: Replicas decreased to: {current_replicas}, system waits 5 seconds")  # Print scale in message
        scale_in(service_name, 1)  # Decrease replicas
    else:
        print(f"Already at minimum replicas: {min_replicas}")  # Indicate already at minimum replicas

    return current_replicas

def scale_out_action(service_name, max_replicas):
    """
    Logic for the scale out action.

    Args:
        service_name (str): Name of the service to scale.
        max_replicas (int): Maximum number of replicas allowed.
        rewards (list): List to store rewards.

    Returns:
        tuple: Tuple containing the updated rewards list and the new replica count.
    """
    # Get current replica count
    current_replicas = get_current_replica_count(service_name)
    
    # If current replicas is not None and less than max replicas
    if current_replicas is not None and current_replicas < max_replicas:
        current_replicas = current_replicas + 1
        print(f"Horizontal Scale Out: Replicas increased to: {current_replicas}")  # Print scale out message
        scale_out(service_name, current_replicas)  # Increase replicas
    else:
        print("Already at maximum replicas. Resetting replicas to 1")  # Indicate already at maximum replicas
        reset_replicas(service_name=service_name)

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
        service_name (_type_): _description_
        scale_out_factor (_type_): _description_
    """
    client = docker.from_env()
    service = client.services.get(service_name)
    current_replicas = get_current_replica_count(service_name)
    desired_replicas = current_replicas - scale_out_factor
    service.scale(desired_replicas)
    time.sleep(cooldownTimeInSec)

def get_reward(cpu_value, ram_value, cpu_threshold, ram_threshold):
    """
    Calculates the reward based on CPU and RAM values.

    Args:
        cpu_value (float): The CPU value.
        ram_value (float): The RAM value.
        cpu_threshold (float): The CPU threshold.
        ram_threshold (float): The RAM threshold.

    Returns:
        int: The calculated reward.
    """
    if cpu_value is not None and ram_value is not None:
        cpu_threshold_10_percent = cpu_value * 0.05
        ram_threshold_10_percent = ram_value * 0.05
        cpu_threshold_merged = cpu_threshold + cpu_threshold_10_percent
        ram_threshold_merged = ram_threshold + ram_threshold_10_percent
        print(f"CPU_Plus_10%: {cpu_threshold_merged} RAM_Plus_10%:{ram_threshold_merged}")
        if cpu_value <= cpu_threshold_merged and ram_value <= ram_threshold_merged:
            print(f"Reward={20}, cpu_value={cpu_value} <= {cpu_threshold_merged} and ram_value={ram_value} <= {ram_threshold_merged}")
            return 20
        else:
            print(f"Reward={-10}, cpu_value={cpu_value} >= {cpu_threshold_merged} and ram_value={ram_value} >= {ram_threshold_merged}")
            return -10
    else:
        return 0

def fetch_data():
    """Fetching the data from the API

    Returns:
        _type_: cpu_percent(int), ram_percent(int), time_up(int) or None for all.
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
    Reset the service to one replica.
    Args:
        service_name (_type_): _description_
    """
    client = docker.from_env()
    service = client.services.get(service_name)
    desired_replicas = 1
    service.scale(desired_replicas)
    time.sleep(cooldownTimeInSec)

def Calculate_Thresholds():
    """
    Calculates the CPU and RAM thresholds.

    Returns:
        tuple: A tuple containing the CPU and RAM thresholds.
    """
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is not None:
        cpu_threshold = 15 + (current_replicas - 1) * 4
        ram_threshold = 15 + (current_replicas - 1) * 5 #if current_replicas <= 0 else 101
    else:
        cpu_threshold = 0  # Default value if replicas count is not available
        ram_threshold = 10  # Default value if replicas count is not available

    print(f"Thresholds calculated as CPU:{cpu_threshold}, RAM: {ram_threshold}")
    return cpu_threshold, ram_threshold

url = 'http://prometheus:9090/api/v1/query'

class AutoscaleEnv(gym.Env):
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
        # Reset environment to initial state
        reset_replicas(service_name = service_name)  # Reset Env.
        reset_replicas(service_name= service_produce_load_name)
        return self._get_observation()

    def step(self, action):
        # Take action and observe new state and reward
        if action == 0:  # scale_out
            scale_out_action(service_name=self.service_name, max_replicas=self.max_replicas)
            # scale_out_action(service_name=service_produce_load_name, max_replicas=1)
            self.cpu_threshold, self.ram_threshold = Calculate_Thresholds()
            
        elif action == 1:  # scale_in
            scale_in_action(service_name=self.service_name, min_replicas=self.min_replicas)
            # scale_in_action(service_name=service_produce_load_name, min_replicas=1)
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
        while True:
            # Return the current CPU and RAM values as the observation
            tuple_data = fetch_data()
            # Check if data is not None and not empty
            if tuple_data is not None and any(ele is not None for ele in tuple_data):
                cpu_value, ram_value, _ = tuple_data  # Implement your metric fetching logic here
                return np.array([cpu_value, ram_value], dtype=np.float32)
            
            # If data is None or empty, wait for a short duration before trying again
            time.sleep(cooldownTimeInSec)  # Adjust the duration based on your requirements

    def _is_done(self):
        elapsed_time_minutes = (time.time() - self.start_time) / 60.0
        return elapsed_time_minutes >= self.max_time_minutes