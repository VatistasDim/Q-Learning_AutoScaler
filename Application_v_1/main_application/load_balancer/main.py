"""
Autoscaler Control Script

This script implements a control loop that monitors CPU and RAM metrics, takes certain actions based on thresholds and conditions,
and updates a Q-value. It also handles scaling operations based on the current state and actions.

Constants:
- url: Prometheus URL for metric retrieval.
- service_name: The name of the observable service.
- application_url: URL for application training.
- cpu_threshold: CPU threshold for scaling operations.
- ram_threshold: RAM threshold for scaling operations.
- max_replicas: Maximum number of replicas.
- min_replicas: Minimum number of replicas.
- num_states: Number of discretized states.
- Q: Q-value array for Q-learning.
- iteration: Current iteration count.

Functions:
- discretize_state(cpu_value, ram_value): Discretizes CPU and RAM values.
- select_action(Q, cpu_state, ram_state): Selects an action based on the given state.
- update_q_value(Q, state, action, reward, next_state): Updates the Q-value based on the given state, action, reward, and next state.
- fetch_data(): Fetches CPU, RAM, and running time metrics from Prometheus.
- calculate_mse(Q, target_values): Calculates Mean Squared Error (MSE).
- plot_values(iterations, mse_values, save_path): Plots MSE over iterations.

Usage:
- Set the necessary constants for metric retrieval, service details, thresholds, and Q-learning.
- Configure the training steps, save path, and validation interval.
- Run the script to perform Q-learning-based autoscaling.
"""

import numpy as np
import random, logging
import prometheus_metrics
import matplotlib.pyplot as plt
from datetime import datetime
from autoscaler_env import AutoscaleEnv
from docker_api import DockerAPI
from costs import Costs
import docker

url = 'http://prometheus:9090/api/v1/query'
service_name = 'mystack_application'
application_url = 'http://application:8501/train'
max_replicas = 10
min_replicas = 1
max_cpu_shares = 100
num_states = 3
w_perf = 0.5 # represents the weight assigned to the performance penalty term in the cost function
w_res = 0.4 # w_perf, it's a constant value that determines the importance or impact of the resource cost in the overall cost calculation.
#Q_file = "q_values.npy"
#Q = np.load(Q_file) if Q_file else np.zeros((num_states, num_states, 2))
# Define the maximum values for each state variable
max_ui = 100  # Maximum value for ui
max_ci = 1024  # Maximum value for ci
max_ki = 10  # Maximum value for ki
# Initialize the Q array with zeros
Q = np.zeros((max_ui + 1, max_ci + 1, max_ki + 1))
iteration = 1

def discretize_state(cpu_value):
    """
    Discretizes CPU and RAM values.

    Args:
        cpu_value (float): The CPU value.
        ram_value (float): The RAM value.

    Returns:
        tuple: A tuple containing the discretized CPU and RAM states.
    """
    cpu_state = int(cpu_value)
    
    cpu_state = max(0, min(cpu_state, num_states - 1))

    return cpu_state

def select_action(Q, epsilon):
    """
    Select an action using an epsilon-greedy strategy.

    Parameters:
        Q (numpy.ndarray): The Q-values.
        epsilon (float): The probability of choosing a random action.

    Returns:
        int: The selected action (0 or 1).
    """
    if np.random.uniform(0, 1) < epsilon:
        # Randomly choose between 0 and 1
        return np.random.randint(2)
    else:
        # Choose the greedy action based on Q-values
        greedy_action = np.argmax(Q)
        return min(greedy_action, 1)  # Ensure the action is within [0, 1]

def update_q_value(Q, reward, alpha, gamma, current_state, next_state, current_action, next_action):
    """
    Update the Q-value based on the given current state, action, reward, and next state.

    Parameters:
        Q (numpy.ndarray): The Q-values.
        reward (float): The reward received for the action.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        current_state (tuple): The current state (ui, ci, ki).
        next_state (tuple): The next state (ui_next, ci_next, ki_next).
        current_action (tuple): The selected current action (a1, a2).
        next_action (tuple): The selected next action (a1_next, a2_next).

    Returns:
        numpy.ndarray: Updated Q-values.
    """
    ui, ci, ki = current_state
    ui_next, ci_next, ki_next = next_state
    ui = round(ui)
    ci = round(ci)
    ki = round(ki)
    ui_next = round(ui_next)
    ci_next = round(ci_next)
    ki_next = round(ki_next)
    print(f'ui:{ui}')
    print(f'ci:{ci}')
    print(f'ki:{ki}')
    print(f'ui_next:{ui_next}')
    print(f'ci_next:{ci_next}')
    print(f'ki_next:{ki_next}')
    # Calculate Q(si+1, a') using the Q-values for the next state and action
    next_q_value = Q[ui_next, ci_next, ki_next]
    # Update Q-value for the given state and action
    Q[ui][ci][ki] = \
        (1 - alpha) * Q[ci][ci_next][ki_next] + \
        alpha * (reward + gamma * next_q_value)
    return Q

def normalize_cpu_shares(cpu_shares):
    inner_list = cpu_shares[0]
    total_shares = sum(inner_list)
    normalized_cpu_shares = total_shares / len(inner_list)
    return normalized_cpu_shares

def state_to_index(state):
    """
    Convert the state tuple to an index for accessing Q-values.

    Parameters:
        state (tuple): A tuple representing the state (ki, ui, ci).

    Returns:
        tuple: A tuple representing the index.
    """
    # Convert the state tuple to an index
    # Here, you can implement a mapping logic based on the ranges of ki, ui, ci
    # For simplicity, let's assume ki, ui, ci are already discretized and their values are indices
    return state

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
                response_time = int(float(metrics[3]))
                return cpu_percent, ram_percent, time_up, response_time
        return None, None, None
    except Exception as e:
        print("An error occurred during service metrics retrieval:", e)
        return None, None, None

def calculate_mse(Q, env, observations):
    mse_sum = 0

    # Iterate over observations obtained during training
    for observation in observations:
        num_containers_obs, cpu_value_obs, cpu_share_obs = observation

        # Discretize observed values
        discretized_num_containers_obs = Discretizer.discretize_num_containers(num_containers_obs, max_num_containers, num_states=self.num_states)
        discretized_cpu_share_obs = Discretizer.discretize_stack_containers_cpu_shares(cpu_value_obs, max_cpu_shares, self.num_states)

        # Retrieve corresponding Q-value from Q-table
        Q_value = Q[discretized_num_containers_obs, cpu_share_obs]

        # Calculate squared difference and accumulate
        mse_sum += (Q_value - cpu_value_obs) ** 2

    # Calculate mean squared error
    mse = mse_sum / len(observations)
    return mse

def plot_mse_values(iterations, mse_values, save_path):
    plt.plot(iterations, mse_values, label='MSE')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Value')
    plt.title('MSE Over Iterations')
    plt.legend()
    plt.savefig(f'{save_path}/mse_plot_iteration_{iteration}.png')

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
    
def check_container_change(before_containers, after_containers):
    """
    Check the change in the number of containers before and after an action.

    Parameters:
        before_containers (int): The number of containers before the action.
        after_containers (int): The number of containers after the action.

    Returns:
        int: A binary value indicating the change:
             - 1 if the number of containers increased.
             - 0 if the number of containers stayed the same or decreased.
    """
    if after_containers > before_containers:
        return 1
    elif after_containers < before_containers:
        return 0
    else:
        return 0
    
def check_cpu_share_change(before_cpu_shares, after_cpu_shares):
    """
    Check the change in CPU shares before and after an action for each container.

    Parameters:
        before_cpu_shares (list): List of lists representing CPU shares before the action.
        after_cpu_shares (list): List of lists representing CPU shares after the action.

    Returns:
        list: List of lists representing changes in CPU shares for each container.
    """
    cpu_share_changes = []
    for before_container, after_container in zip(before_cpu_shares, after_cpu_shares):
        changes = []
        for before_share, after_share in zip(before_container, after_container):
            change = after_share - before_share
            changes.append(change)
        cpu_share_changes.append(changes)
    print(f'cpu_share_changes:{cpu_share_changes}')
    return cpu_share_changes

def normalize_cpu_share_change(cpu_share_changes):
    """
    Normalize the changes in CPU shares to binary values indicating increase or decrease.

    Parameters:
        cpu_share_changes (list): List of lists representing changes in CPU shares for each container.

    Returns:
        list: List of lists representing normalized changes in CPU shares for each container.
    """
    normalized_changes = []
    for container_changes in cpu_share_changes:
        container_normalized_changes = []
        for change in container_changes:
            if change > 0:
                container_normalized_changes.append(1)  # Increase
            else:
                container_normalized_changes.append(0)  # No change or decrease
        normalized_changes.append(container_normalized_changes)
    print(f'normalized_changes: {normalized_changes}')
    return normalized_changes

def state():
    """
    Retrieve the current state of the system, including CPU utilization, CPU shares of containers,
    and the number of running containers.

    Returns:
        tuple: A tuple containing the following elements:
            - dict: A dictionary containing the CPU shares of all containers in the stack,
                    with container names as keys and CPU shares as values.
            - float: The current CPU utilization of the system.
            - int: The number of running containers in the specified service.
    """
    docker_client = DockerAPI(service_name)
    cpu_value, _, _, _ = fetch_data()
    c_cpu_shares = docker_client.get_stack_containers_cpu_shares(service_name=service_name)
    u_cpu_utilzation = cpu_value
    k_running_containers = get_current_replica_count(service_prefix=service_name)
    return c_cpu_shares, u_cpu_utilzation, k_running_containers

def binarize_cpu_shares(cpu_shares):
    """
    Binarize CPU shares values.

    Parameters:
        cpu_shares (dict): Dictionary containing container IDs as keys and CPU shares lists as values.

    Returns:
        list: List of binarized CPU shares lists.
    """
    binarized_cpu_shares = []
    for shares_list in cpu_shares.values():
        binarized_shares = [1 if share > 0 else 0 for share in shares_list]
        binarized_cpu_shares.append(binarized_shares)
    return binarized_cpu_shares

if __name__ == "__main__":
    """
    The main method for this project
    """
    train_steps = 101
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Application Information:\nStart date & time:{dt_string}\nObservable service name:{service_name}\nContext urls:{url}, {application_url},\nTrain Steps = {train_steps}\n")
    mse_values = [] # Initialize empty list to store MSE values
    rewards = []
    replicas_count = []
    save_path = '/plots' # Set the save path inside the container
    validation_interval = 101 # Perform validation every 101 iterations
    env = AutoscaleEnv(service_name, min_replicas, max_replicas, num_states)
    obs = env.reset()
    wperf = 0.90
    wres = 0.09
    wadp = 0.01
    Rmax = 0.05
    alpha = 0.5
    gamma = 0.5
    epsilon = 0.4
    cres = 0.01
    k_running_containers_next_state = 0
    c_cpu_shares_next_state = None
    mse_values = []
    for iteration in range(1, train_steps):  # Run iterations
        logger = logging.getLogger(__name__)  # Initialize a logger
        print(f"\n--------Iteration No:{iteration}")  # Print the current iteration number
        cpu_value, _, time_running, response_time = fetch_data()  # Get CPU, RAM values and a placeholder value
        R = response_time
        print(f"Metrics: |CPU:{str(cpu_value)}% |Time running:{str(time_running)}s")  # Print metrics
        observation = env._get_observation()
        
        c_cpu_shares, u_cpu_utilzation, k_running_containers = state() # read the current state
        current_normalized_cpu_shares = binarize_cpu_shares(c_cpu_shares)
        a1_current_state = check_container_change(k_running_containers, k_running_containers_next_state) # the value of a1 (increased/decreased containers)
        
        if c_cpu_shares_next_state is None:
            cpu_share_changes = current_normalized_cpu_shares
        else:
            cpu_share_changes = check_cpu_share_change(c_cpu_shares.values(), c_cpu_shares_next_state.values()) # Calculate a2 (CPU share change)
        
        action = select_action(Q, epsilon) # Take action based on observation
        print(f'action:{action}')
        
        a2_current_state = normalize_cpu_share_change(cpu_share_changes) # Normalize a2 in interval of [0,1] in CPU shares changes
        current_state, cost, done, _ = env.step(action) # Take a step in the environment
        c_cpu_shares_next_state, u_cpu_utilzation_next_state, k_running_containers_next_state = state() # read the next current state
        print(f'running_containers:{k_running_containers}')
        print(f'running_containers_next_state:{k_running_containers_next_state}')
        a1_next_state = check_container_change(k_running_containers, k_running_containers_next_state) # the value of a1 (increased/decreased containers)
        cpu_share_changes = check_cpu_share_change(c_cpu_shares.values(), c_cpu_shares_next_state.values()) # Calculate a2 (CPU share change)
        a2_next_state = normalize_cpu_share_change(cpu_share_changes) # Normalize a2 in interval of [0,1] in CPU shares changes
        next_state = env._get_observation() # Observe the next state after the action is taken
        normalized_c_cpu_shares_next_state = binarize_cpu_shares(c_cpu_shares_next_state)
        
        cost = Costs
        total_cost = cost.overall_cost_function(wadp, wres, wperf, k_running_containers, a1_next_state, u_cpu_utilzation, a2_next_state, Rmax, 1, 10, cres, R)
        total_cost = round(total_cost, 5)
        current_normalized_cpu_shares = normalize_cpu_shares(current_normalized_cpu_shares)
        normalized_c_cpu_shares_next_state = normalize_cpu_shares(normalized_c_cpu_shares_next_state)
        a2_current_state = normalize_cpu_shares(a2_current_state)
        a2_next_state = normalize_cpu_shares(a2_next_state)
        current_state = (u_cpu_utilzation, current_normalized_cpu_shares, k_running_containers)
        next_state = (u_cpu_utilzation_next_state, normalized_c_cpu_shares_next_state, k_running_containers_next_state)
        current_state_action = (a1_current_state, a2_current_state)
        next_state_action = (a1_next_state, a2_next_state)
        print(Q.shape)
        print(f'total_cost:{total_cost}')
        print(f'alpha:{alpha}')
        print(f'gamma:{gamma}')
        print(f'current_state:{current_state}')
        print(f'next_state:{next_state}')
        print(f'current_state_action:{current_state_action}')
        print(f'next_state_action:{next_state_action}')
        update_q_value(Q, total_cost, alpha, gamma, current_state, next_state, current_state_action, next_state_action)
        iteration += 1
        # Log Q-values & Log rewards
        # print(f"Q-values: \n{Q}")
        
        # Calculate MSE and store in list
        #mse = calculate_mse(Q, observation)
        #mse_values.append(mse)
        
        # if iteration % 50 == 0:  # Save plot every 50 iterations
        #     print("Plotting...")
        #     plot_mse_values(range(1, iteration - 1), mse_values, save_path)
        # np.save('/QSavedWeights/q_values.npy', Q)
    else:  # If CPU or RAM values are None
        print("No metrics available, wait...")  # Indicate no metrics available
