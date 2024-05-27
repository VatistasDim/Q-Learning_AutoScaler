import numpy as np
import random, logging, time
import prometheus_metrics
import matplotlib.pyplot as plt
from datetime import datetime
from autoscaler_env import AutoscaleEnv
from docker_api import DockerAPI
from costs import Costs
import docker
import itertools

wperf = 0.90
wres = 0.09
wadp = 0.01
Rmax = 0.05
alpha = 0.5
gamma = 0.5
epsilon = 0.4
cres = 0.01
wait_time = 15
url = 'http://prometheus:9090/api/v1/query'
service_name = 'mystack_application'
application_url = 'http://application:8501/train'
max_replicas = 10
min_replicas = 1
max_cpu_shares = 100
num_states = 3
w_perf = 0.5 # represents the weight assigned to the performance penalty term in the cost function
w_res = 0.4 # w_perf, it's a constant value that determines the importance or impact of the resource cost in the overall cost calculation.
max_containers = 11
# Define the ranges for CPU utilization, number of running containers, and CPU shares
cpu_utilization_values = range(101)  # CPU utilization values from 0 to 100
k_range = range(1, max_containers)  # Number of running containers (1 to 11)
cpu_shares_values = [1.0, 0.75, 0.50, 0.25, 0.5]  # CPU shares (1.0, 0.75, 0.50, 0.25, 0.5)

# Generate all combinations of CPU utilization, K, and CPU shares
state_space = list(itertools.product(cpu_shares_values, cpu_utilization_values, k_range))
action_space = [-1, 0, 1]  # Actions: -1 (scale in), 0 (do nothing), 1 (scale out)
Q = np.zeros((len(state_space), len(action_space)))
iteration = 1

def transition(action):
    running_containers = get_current_replica_count(service_prefix = service_name)
    if action == -1 and running_containers > 1:  # Scale in (decrease containers)
        scale_in(service_name = service_name, scale_out_factor = 1)
    elif action == 1:  # Scale out (increase containers)
        desired_replicas = get_current_replica_count(service_prefix = service_name)
        if desired_replicas < 10:
            desired_replicas = desired_replicas + 1
        else:
            print("Max replicas.")
            c, u, k = state()
            return (c, u, k)
        scale_out(service_name = service_name, desired_replicas = desired_replicas)
    elif action == 0:
        time.sleep(15)
    c, u, k = state()
    return (c, u, k)

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
        greedy_action = np.argmin(Q)
        print(f'greedy_action:{greedy_action}')
        return min(greedy_action, 1)  # Ensure the action is within [0, 1]

#def update_q_value(Q, reward, alpha, gamma, current_state, next_state, current_action, next_action):
#    """
#    Update the Q-value based on the given current state, action, reward, and next state.
#
#    Parameters:
#        Q (numpy.ndarray): The Q-values.
#        reward (float): The reward received for the action.
#        alpha (float): The learning rate.
#        gamma (float): The discount factor.
#        current_state (tuple): The current state (ui, ci, ki).
#        next_state (tuple): The next state (ui_next, ci_next, ki_next).
#        current_action (tuple): The selected current action (a1, a2).
#        next_action (tuple): The selected next action (a1_next, a2_next).
#
#    Returns:
#        numpy.ndarray: Updated Q-values.
#    """
#    ui, ci, ki = current_state
#    ui_next, ci_next, ki_next = next_state
#    a1, a2 = current_action
#    a1_next, a2_next = next_action
#    ui = round(ui)
#    ci = round(ci)
#    ki = round(ki)
#    ui_next = round(ui_next)
#    ci_next = round(ci_next)
#    ki_next = round(ki_next)
#    a1 = round(a1)
#    a2 = round(a2)
#    a1_next = round(a1_next)
#    a2_next = round(a2_next)
#    print(f'ui:{ui}')
#    print(f'ci:{ci}')
#    print(f'ki:{ki}')
#    print(f'ui_next:{ui_next}')
#    print(f'ci_next:{ci_next}')
#    print(f'ki_next:{ki_next}')
#    print(f'a1:{a1}')
#    print(f'a2:{a2}')
#    print(f'a1_next:{a1_next}')
#    print(f'a2_next:{a2_next}')
#    # Calculate Q(si+1, a') using the Q-values for the next state and action
#    next_q_value = Q[ui_next, ci_next, ki_next, a1_next, a2_next]
#    # Update Q-value for the given state and action and the next_q_value
#    Q[ui][ci][ki][a1][a2] = (1 - alpha) * Q[ci][ci][ki][a1][a2] + alpha * (reward + gamma * next_q_value)
#    return Q

def fetch_data():
    """Fetching the data from the API

    Returns:
        _type_: cpu_percent(int), ram_percent(int), time_up(int) or None for all.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            cpu_percent, ram_percent, time_up, response_time, cpu_shares = prometheus_metrics.start_metrics_service(url=url)
            if time_up != '0':
                cpu_percent = int(float(cpu_percent))
                ram_percent = int(float(ram_percent))
                time_up = int(float(time_up))
                response_time = int(float(response_time))
                cpu_shares = int(cpu_shares)
                cpu_shares = calculate_cpu_shares(cpu_shares)
                # Check if any value is None, if so, continue fetching
                if None in (cpu_percent, ram_percent, time_up, response_time, cpu_shares):
                    continue
                return cpu_percent, ram_percent, time_up, response_time, cpu_shares
        except Exception as e:
            print(f"An error occurred during service metrics retrieval (Attempt {attempt + 1}/{max_attempts}):", e)
            if attempt < max_attempts - 1:
                # Wait for a few seconds before retrying
                time.sleep(60)
    print("Failed to retrieve metrics after multiple attempts.")
    return None, None, None, None, None


def calculate_cpu_shares(cpu_share):
    if cpu_share == 1024:
        return 1
    elif cpu_share == 512:
        return 0.5
    elif cpu_share == 256:
        return 0.25
    elif cpu_share == 128:
        return 0.15
    else:
        return 0.5

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

def scale_out(service_name, desired_replicas):
    """
    Scales out a service to the specified number of replicas.
    """
    client = docker.from_env()
    service = client.services.get(service_name)
    service.scale(desired_replicas)
    print(f"Service '{service_name}' scaled to {desired_replicas} replicas.")
    time.sleep(15)

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
    time.sleep(15)

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
    cpu_value, _, _, _, c_cpu_shares = fetch_data()
    u_cpu_utilzation = cpu_value
    k_running_containers = get_current_replica_count(service_prefix=service_name)
    return c_cpu_shares, u_cpu_utilzation, k_running_containers

#if __name__ == "__main__":
#    """
#    The main method for this project
#    """
#    train_steps = 101
#    now = datetime.now()
#    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#    print(f"Application Information:\nStart date & time:{dt_string}\nObservable service name:{service_name}\nContext urls:{url}, {application_url},\nTrain Steps = {train_steps}\n")
#    #mse_values = [] # Initialize empty list to store MSE values
#    #rewards = []
#    #replicas_count = []
#    #save_path = '/plots' # Set the save path inside the container
#    #validation_interval = 101 # Perform validation every 101 iterations
#    #env = AutoscaleEnv(service_name, min_replicas, max_replicas, num_states)
#    #obs = env.reset()
#    wperf = 0.90
#    wres = 0.09
#    wadp = 0.01
#    Rmax = 0.05
#    alpha = 0.5
#    gamma = 0.5
#    epsilon = 0.4
#    cres = 0.01
#    k_running_containers_next_state = 0
#    c_cpu_shares_next_state = None
#    mse_values = []
#    for iteration in range(1, train_steps):  # Run iterations
#        logger = logging.getLogger(__name__)  # Initialize a logger
#        print(f"\n--------Iteration No:{iteration}")  # Print the current iteration number
#        cpu_value, _, time_running, response_time, cpu_shares = fetch_data()  # Get CPU, RAM values and a placeholder value
#        R = response_time
#        print(f"Metrics: | CPU:{str(cpu_value)}% | Time Running:{str(time_running)}s | Response Time:{str(response_time)} | CPU Shares: {str(cpu_shares)}")  # Print metrics
#        c_cpu_shares, u_cpu_utilzation, k_running_containers = state() # read the current state
#        # action = select_action(Q, epsilon) # Take action based on observation
#        # Select action using epsilon-greedy strategy
#        if np.random.uniform(0, 1) < epsilon:
#            action = np.random.choice(action_space)
#        else:
#            action = np.argmin(Q[state_space.index(state)])
#        print(f'action:{action}')
#        next_state = transition(state, action)
#        a2_current_state = c_cpu_shares
#        current_state, cost, done, _ = env.step(action) # Take a step in the environment
#        print(f'System waits for {wait_time} seconds to retrieve the next state.')
#        time.sleep(wait_time)
#        c_cpu_shares_next_state, u_cpu_utilzation_next_state, k_running_containers_next_state = state() # read the next current state
#        a2_next_state = c_cpu_shares_next_state
#        next_state = env._get_observation() # Observe the next state after the action is taken
#        cost = Costs
#        total_cost = cost.overall_cost_function(wadp, 
#                                                wres, 
#                                                wperf, 
#                                                k_running_containers_next_state, 
#                                                u_cpu_utilzation_next_state,
#                                                c_cpu_shares_next_state, 
#                                                action, 
#                                                k_running_containers_next_state,
#                                                a2_next_state, 
#                                                Rmax,
#                                                10, 
#                                                R)        
#        total_cost = round(total_cost, 5)
#        Q[state_space.index(state)][action_space.index(action)] += alpha * \
#                (total_cost + gamma * np.max(Q[state_space.index(next_state)]) - Q[state_space.index(state)][action_space.index(action)])
#        
#        #a2_current_state = c_cpu_shares
#        #a2_next_state = c_cpu_shares_next_state
#        #current_state = (u_cpu_utilzation, a2_current_state, k_running_containers)
#        #next_state = (u_cpu_utilzation_next_state, a2_next_state, k_running_containers_next_state)
#        #current_state_action = (k_running_containers, a2_current_state)
#        #next_state_action = (k_running_containers_next_state, a2_next_state)
#        #print(f'total_cost:{total_cost}')
#        #print(f'alpha:{alpha}')
#        #print(f'gamma:{gamma}')
#        #print(f'current_state:{current_state}')
#        #print(f'next_state:{next_state}')
#        #print(f'current_state_action:{current_state_action}')
#        #print(f'next_state_action:{next_state_action}')
#        #update_q_value(Q, total_cost, alpha, gamma, current_state, next_state, current_state_action, next_state_action)
#        #max_cost_state = np.unravel_index(np.argmax(Q), Q.shape)
#        #max_cost = np.max(Q)
#        #print(f'Highest Cost State: {max_cost_state}')
#        #print(f'Highest Cost: {max_cost}')
#        #min_cost_state = np.unravel_index(np.argmin(Q), Q.shape)
#        #min_cost = np.min(Q)
#        #print(f'Lowest Cost State: {min_cost_state}')
#        #print(f'Min Cost: {min_cost}')
#        #iteration += 1
#    else:  # If CPU or RAM values are None
#        print("No metrics available, wait...")  # Indicate no metrics available

def find_nearest_state(state, state_space):
    """
    Find the nearest matching state in the state space.

    Parameters:
        state (tuple): The state to match.
        state_space (list): The list of states in the state space.

    Returns:
        tuple: The nearest matching state.
    """
    distances = [sum(abs(np.array(state) - np.array(s))) for s in state_space]
    nearest_index = np.argmin(distances)
    return state_space[nearest_index]

import numpy as np
import matplotlib.pyplot as plt

def q_learning(num_episodes):
    episode = 0  # Initialize episode counter
    costs_per_episode = []  # List to store total cost per episode
    total_time_per_episode = []  # List to store total response time per episode
    average_cost_per_episode = []  # List to store average reward per episode

    while episode < num_episodes:
        app_state = state()  # Initial state
        total_cost = 0  # Initialize total cost for this episode
        total_time = 0  # Initialize total time for this episode
        total_reward = 0  # Initialize total reward for this episode
        steps = 0  # Track the number of steps per episode

        while True:
            print(f'app_state: {app_state}')
            nearest_state = find_nearest_state(app_state, state_space)
            print(f'nearest_state: {nearest_state}')
            # Select action using epsilon-greedy strategy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(action_space)
            else:
                action = np.argmin(Q[state_space.index(nearest_state)])
                action = np.clip(action, -1, 1)
            print(f'action: {action}')
            # Simulate state transition
            next_state = transition(action)
            print(f'next_state: {next_state}')
            c_cpu_shares_next_state, u_cpu_utilzation_next_state, k_running_containers_next_state = next_state # read the next current state
            a2_next_state = c_cpu_shares_next_state
            cost = Costs
            _, _, _, response_time, _ = fetch_data()  # Get CPU, RAM values and a placeholder value
            R = response_time
            step_cost = cost.overall_cost_function(wadp, 
                                                   wres, 
                                                   wperf, 
                                                   k_running_containers_next_state, 
                                                   u_cpu_utilzation_next_state,
                                                   c_cpu_shares_next_state, 
                                                   action, 
                                                   k_running_containers_next_state,
                                                   a2_next_state, 
                                                   Rmax,
                                                   10, 
                                                   R)      
            step_cost = round(step_cost, 5)
            print(f'Response_time: {R}, Cost of action: {step_cost}')
            total_cost += step_cost  # Accumulate the cost
            total_time += R  # Accumulate the reward
            steps += 1  # Increment step count

            # Update Q-value using Bellman equation
            Q[state_space.index(nearest_state)][action_space.index(action)] += alpha * \
                (step_cost + gamma * np.min(Q[state_space.index(next_state)]) - Q[state_space.index(nearest_state)][action_space.index(action)])
            # Move to next state
            app_state = next_state
            print(f'Episode: {episode + 1}')
            # Check if the current episode should end
            if app_state[2] == 9:  # Terminal state reached (maximum number of instances)
                break

        costs_per_episode.append(total_cost)  # Store total cost for this episode
        total_time_per_episode.append(total_time)  # Store response for this episode
        average_cost_per_episode.append(total_cost / steps)  # Store average reward for this episode
        episode += 1

    return costs_per_episode, total_time_per_episode, average_cost_per_episode

# Run Q-learning
num_episodes = 1000
epsilon = 0.4  # Epsilon for epsilon-greedy strategy
costs, total_time, avg_cost = q_learning(num_episodes)

# Visualize the data
plt.figure(figsize=(18, 6))

# Plot total cost per episode
plt.subplot(1, 3, 1)
plt.plot(range(num_episodes), costs, label='Total Cost')
plt.xlabel('Episode')
plt.ylabel('Total Cost')
plt.title('Total Cost per Episode')
plt.legend()

# Plot total reward per episode
plt.subplot(1, 3, 2)
plt.plot(range(num_episodes), total_time, label='Total Time')
plt.xlabel('Episode')
plt.ylabel('Total Time')
plt.title('Total Time per Episode')
plt.legend()

# Plot average reward per episode
plt.subplot(1, 3, 3)
plt.plot(range(num_episodes), avg_cost, label='Average Cost')
plt.xlabel('Episode')
plt.ylabel('Average Cost')
plt.title('Average Cost per Episode')
plt.legend()

# Save the figure to the mounted volume directory
plt.tight_layout()
plt.savefig('/plots/q_learning_performance.png')
plt.close()

# Print learned Q-values
print("Learned Q-values:")
print(Q)

np.save('/QSavedWeights/Q_table.npy', Q)

for state_idx, q_values in enumerate(Q):
    favored_action = np.argmax(q_values)
    print(f"For state {state_idx}, favored action is {favored_action}")
