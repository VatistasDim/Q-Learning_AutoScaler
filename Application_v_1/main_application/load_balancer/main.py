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
Q = np.zeros((num_states, num_states, 3))
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

def select_action(Q, observation):
    epsilon = 0.4
    discretized_num_containers, cpu_value, discretized_cpu_share = observation

    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2])
    else:
        # Discretize the continuous values to use them as indices in the Q-table
        cpu_state = min(max(discretized_num_containers, 0), num_states - 1)
        cpu_value_state = min(max(discretize_state(cpu_value), 0), num_states - 1)
        # Choose the action with the highest Q-value
        return np.argmax(Q[cpu_state, cpu_value_state, discretized_cpu_share])


def update_q_value(Q, state, action, cost, next_state, num_states):
    """
    Updates the Q-value based on the given state, action, reward, and next state.

    Args:
        Q (numpy.ndarray): The Q-values.
        state (tuple): A tuple representing the current state.
        action (int): The selected action.
        reward (int): The reward received for the action.
        next_state (tuple): A tuple representing the next state.
        num_states (int): Number of discretized states.
    """
    alpha = 0.2
    gamma = 0.1

    # Discretize the current state
    cpu_state, cpu_value, _ = state
    cpu_state = min(max(cpu_state, 0), num_states - 1)
    cpu_value_state = discretize_state(cpu_value)
    state = (cpu_state, cpu_value_state)

    # Discretize the next state
    next_cpu_state, next_cpu_value, _ = next_state
    next_cpu_state = min(max(next_cpu_state, 0), num_states - 1)
    next_cpu_value_state = discretize_state(next_cpu_value)
    next_state = (next_cpu_state, next_cpu_value_state)

    # Update Q-value based on the action
    if action == 2:
        Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (cost - Q[state[0], state[1], action])
    else:
        Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (
            cost + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action]
        )
    return Q

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

def calculate_mse(Q, target_values):
    """
    Calculates Mean Squared Error (MSE).

    Args:
        Q (numpy.ndarray): The Q-values.
        target_values (numpy.ndarray): The target values.

    Returns:
        float: The calculated MSE.
    """
    mse = ((Q - target_values)**2).mean()
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
# This code implementing a control loop that monitors CPU and RAM metrics, takes certain actions based on thresholds and conditions, and updates a Q-value. 
# It also handles scaling operations based on the current state and actions.
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
    docker_client = DockerAPI(service_name)
    obs = env.reset()
    for iteration in range(1, train_steps):  # Run iterations
        
        logger = logging.getLogger(__name__)  # Initialize a logger
        print(f"\n--------Iteration No:{iteration}")  # Print the current iteration number
        cpu_value, _, time_running, response_time = fetch_data()  # Get CPU, RAM values and a placeholder value
        print(f"Metrics: |CPU:{str(cpu_value)}% |Time running:{str(time_running)}s")  # Print metrics
        
        # #Observe current state
        # observation = env._get_observation()
        
        # # Take action based on observation
        # action = select_action(Q, observation)
        
        # # Take a step in the environment
        # current_state, cost, done, _ = env.step(action)
        
        # # Observe the next state after the action is taken
        # next_state = env._get_observation()
        
        #RUNNING CONTAINERS
        c_cpu_shares = docker_client.get_stack_containers_cpu_shares(service_name=service_name)
        all_cpu_shares_sets = c_cpu_shares.values()
        all_cpu_shares_list = [value for sublist in all_cpu_shares_sets for value in sublist]
        print(f'Cpu-Shares: {all_cpu_shares_list}')
        average_cpu_shares = sum(all_cpu_shares_list) / len(all_cpu_shares_list)
        print(f'average cpu shares: {average_cpu_shares}')
        max_c_cpu_shares = max(all_cpu_shares_list)
        print(f"Maximum CPU shares: {max_c_cpu_shares}")
        u_cpu_utilzation = cpu_value
        k_running_containers = get_current_replica_count(service_prefix=service_name)
        cost = Costs
        wperf = 0.90
        wres = 0.09
        wadp = 0.01
        Rmax = 0.05
        R = response_time
        cres = 0.01
        total_cost = cost.overall_cost_function(wadp, wres, wperf, k_running_containers, 0, u_cpu_utilzation, 1, Rmax, max_c_cpu_shares, 10, cres, R)
        print(f'total_cost: {total_cost}')
        #KEEP STATES IN SOME KIND OF VARIABLE FOR FURTHER USAGE.
        
        # Update Q value based on action and next state
        # update_q_value(Q, current_state, action, cost, next_state, num_states=num_states)
        
        # Add one iteration
        iteration += 1
        
        # Log Q-values & Log rewards
        #print(f"Reward: {cost}, Q-values: \n{Q}")
        
        # Calculate MSE and store in list
        #mse = calculate_mse(Q, observation)
        #mse_values.append(mse)
        
        # if iteration % 50 == 0:  # Save plot every 50 iterations
        #     print("Plotting...")
        #     plot_mse_values(range(1, iteration - 1), mse_values, save_path)
        # np.save('/QSavedWeights/q_values.npy', Q)
    else:  # If CPU or RAM values are None
        print("No metrics available, wait...")  # Indicate no metrics available
