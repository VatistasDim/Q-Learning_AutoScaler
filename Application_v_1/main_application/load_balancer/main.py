import numpy as np
import random, logging, time
import prometheus_metrics
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from autoscaler_env import AutoscaleEnv
from docker_api import DockerAPI
from costs import Costs
import docker
import pytz, os
import itertools

# Ensure the directory exists
if not os.path.exists('/app/plots'):
    os.makedirs('/app/plots')

# Constants
timezone = pytz.timezone('Europe/Athens')
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
max_cpu_shares = 1024
num_states = 3
w_perf = 0.5  # Weight assigned to the performance penalty term in the cost function
w_res = 0.4  # Determines the importance of the resource cost in the overall cost calculation
max_containers = 11

# Define the ranges for CPU utilization, number of running containers, and CPU shares
cpu_utilization_values = range(101)  # CPU utilization values from 0 to 100
k_range = range(1, max_containers)  # Number of running containers (1 to 10)
cpu_shares_values = [1024, 512, 256, 128]  # CPU shares (1024, 512, 256, 128)

# Generate all combinations of CPU utilization, K, and CPU shares
state_space = list(itertools.product(cpu_shares_values, cpu_utilization_values, k_range))
action_space = [-1, 0, 1, -512, 512]  # Actions: -1 (scale in), 0 (do nothing), 1 (scale out), -512 (decrease CPU shares), 512 (increase CPU shares)

# Initialize Q-table
Q = np.zeros((len(state_space), len(action_space)))

iteration = 1

def transition(action):
    running_containers = get_current_replica_count(service_prefix=service_name)
    current_cpu_shares = get_current_cpu_shares(service_name)
    
    if action == -1 and running_containers > 1:  # Scale in (decrease containers)
        print("Log: Decrease Container by 1")
        scale_in(service_name=service_name, scale_out_factor=1)
    elif action == 1 and running_containers < max_replicas:  # Scale out (increase containers)
        print("Log: Increase Container by 1")
        desired_replicas = running_containers + 1
        scale_out(service_name=service_name, desired_replicas=desired_replicas)
    elif action == -512 and current_cpu_shares > 128:  # Decrease CPU shares
        print("Log: Decrease CPU shares")
        set_cpu_shares(service_name, max(current_cpu_shares - 512, 128))
    elif action == 512 and current_cpu_shares < 1024:  # Increase CPU shares
        print("Log: Increase CPU shares")
        set_cpu_shares(service_name, min(current_cpu_shares + 512, 1024))
    elif action == 0:
        print("Log: No Action")
        time.sleep(15)
    
    c, u, k = state()
    return (c, u, k)

def select_action(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(action_space)
    else:
        state_idx = state_space.index(state)
        return action_space[np.argmax(Q[state_idx])]

def fetch_data():
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
                if None in (cpu_percent, ram_percent, time_up, response_time, cpu_shares):
                    continue
                return cpu_percent, ram_percent, time_up, response_time, cpu_shares
        except Exception as e:
            print(f"Error: An error occurred during service metrics retrieval (Attempt {attempt + 1}/{max_attempts}):", e)
            if attempt < max_attempts - 1:
                time.sleep(60)
    print("Failed to retrieve metrics after multiple attempts.")
    return None, None, None, None, None

def calculate_cpu_shares(cpu_share):
    if cpu_share == 1024:
        return 1.0
    elif cpu_share == 512:
        return 0.5
    elif cpu_share == 256:
        return 0.25
    elif cpu_share == 128:
        return 0.125
    else:
        return 0.5

def get_current_cpu_shares(service_name):
    client = docker.from_env()
    try:
        service = client.services.get(service_name)
        resources = service.attrs['Spec']['TaskTemplate']['Resources']
        cpu_current_shares = resources.get('Limits', {}).get('NanoCPUs', 0) // 1_000_000_000
        return cpu_current_shares
    except docker.errors.NotFound:
        print("Error: Service not found")
        return 0

def get_node_resources(service_name):
    print("Log: Starting gathering node resources")
    client = docker.APIClient(base_url='unix://var/run/docker.sock')
    try:
        service = client.inspect_service(service_name)
        tasks = client.tasks(filters={'service': service_name})
        if not tasks:
            print(f"Error: No tasks found for service {service_name}")
            return None, None
        if 'NodeID' not in tasks[0]:
            print(f"Error: 'NodeID' not found in task {tasks[0]}")
            return None, None
        node_id = tasks[0]['NodeID']
        node = client.inspect_node(node_id)
        node_resources = node['Description']['Resources']
        print(f'Log: Node Id: {node_id}, Node: {node}, Node Resources: {node_resources}')
        return node_resources['NanoCPUs'], node_resources['MemoryBytes']
    except docker.errors.NotFound:
        print("Error: Service or node not found")
        return None, None

def set_cpu_shares(service_name, cpu_shares):
    client = docker.from_env()
    try:
        print("Log: Setting CPU shares")
        service = client.services.get(service_name)
        resources = service.attrs['Spec']['TaskTemplate']['Resources']
        if 'Limits' not in resources:
            resources['Limits'] = {}
        node_nano_cpus, node_memory_bytes = get_node_resources(service_name)
        if node_nano_cpus is None or node_memory_bytes is None:
            print("Error: Could not get node resources")
            return
        print(f'Log: Node Nano Cpus: {node_nano_cpus}, Node Memory Bytes: {node_memory_bytes}')
        current_cpu_shares = resources['Limits'].get('NanoCPUs', 0)
        print(f'Log: Current CPU Shares: {current_cpu_shares}')
        if current_cpu_shares + (cpu_shares * 1_000_000_000) > node_nano_cpus:
            print("Error: Not enough CPU resources available")
            return
        resources['Limits']['NanoCPUs'] = cpu_shares * 1_000_000_000
        service.update(resources=resources)
        time.sleep(15)
    except docker.errors.NotFound:
        print("Error: Cannot Increase CPU Shares")
        pass

def get_current_replica_count(service_prefix):
    client = docker.from_env()
    try:
        for service in client.services.list():
            if service_prefix in service.name:
                return service.attrs['Spec']['Mode']['Replicated']['Replicas']
        return None
    except docker.errors.NotFound:
        return None

def scale_out(service_name, desired_replicas):
    client = docker.from_env()
    service = client.services.get(service_name)
    service.scale(desired_replicas)
    print(f"Log: Service '{service_name}' scaled to {desired_replicas} replicas.")
    time.sleep(15)

def scale_in(service_name, scale_out_factor):
    client = docker.from_env()
    service = client.services.get(service_name)
    current_replicas = get_current_replica_count(service_name)
    desired_replicas = current_replicas - scale_out_factor
    service.scale(desired_replicas)
    print(f"Log: Service '{service_name}' scaled to {desired_replicas} replicas.")
    time.sleep(15)

def state():
    docker_client = DockerAPI(service_name)
    cpu_value, _, _, _, c_cpu_shares = fetch_data()
    c_cpu_shares = get_current_cpu_shares(service_name)
    u_cpu_utilization = cpu_value
    print(f'Log: CPU Shares Real: {u_cpu_utilization}')
    k_running_containers = get_current_replica_count(service_prefix=service_name)
    return c_cpu_shares, u_cpu_utilization, k_running_containers

def find_nearest_state(state, state_space):
    distances = [sum(abs(np.array(state) - np.array(s))) for s in state_space]
    nearest_index = np.argmin(distances)
    return state_space[nearest_index]

def q_learning(num_episodes):
    episode = 1
    costs_per_episode = []
    total_time_per_episode = []
    average_cost_per_episode = []
    Rmax_violations = []
    average_cpu_utilization = []
    average_cpu_shares = []
    average_num_containers = []
    average_response_time = []
    adaptation_counts = []
    total_episodes = num_episodes

    while episode <= num_episodes:
        app_state = state()
        total_cost = 0
        total_time = 0
        total_reward = 0
        total_cpu_utilization = 0
        total_cpu_shares = 0
        total_containers = 0
        total_response_time = 0
        steps = 0
        adaptation_count = 0
        Rmax_violation_count = 0
        next_state = app_state
        start_time = datetime.now()

        while True:
            current_state = next_state
            nearest_state = find_nearest_state(current_state, state_space)
            action = select_action(Q, nearest_state, epsilon)
            next_state = transition(action)
            _, _, _, performance_penalty, _ = fetch_data()
            if performance_penalty is not None:
                performance_penalty = performance_penalty - 0.50
            else:
                print("Error: Performance penalty is None. Skipping calculation.")
                performance_penalty = 0
            
            resource_cost = cres * float(next_state[2])
            if performance_penalty < 0:
                performance_penalty = 0
            total_cost += w_perf * performance_penalty + w_res * resource_cost
            total_time += (datetime.now() - start_time).total_seconds()
            total_reward += total_cost
            total_cpu_utilization += current_state[1]
            total_cpu_shares += current_state[0]
            total_containers += current_state[2]
            _, _, _, total_response_time, _ = fetch_data()
            steps += 1
            adaptation_count += 1
            if performance_penalty > Rmax:
                Rmax_violation_count += 1

            current_state_idx = state_space.index(nearest_state)
            next_state_idx = state_space.index(find_nearest_state(next_state, state_space))
            Q[current_state_idx, action_space.index(action)] = (1 - alpha) * Q[current_state_idx, action_space.index(action)] + alpha * (total_reward + gamma * min(Q[next_state_idx, :]))
            
            # Calculate ETA
            remaining_episodes = total_episodes - episode
            average_time_per_episode = total_time / episode if episode > 0 else 0
            remaining_time = remaining_episodes * average_time_per_episode
            eta = datetime.now() + timedelta(seconds=remaining_time)
            athens_tz = pytz.timezone('Europe/Athens')
            eta_athens = eta.astimezone(athens_tz)

            print(f"Log: Episode: {episode}, ETA: {eta_athens}")

            if datetime.now() - start_time > timedelta(minutes=1):
                break

        episode += 1
        costs_per_episode.append(total_cost / steps)
        total_time_per_episode.append(total_time)
        average_cost_per_episode.append(total_reward / steps)
        Rmax_violations.append(Rmax_violation_count)
        average_cpu_utilization.append(total_cpu_utilization / steps)
        average_cpu_shares.append(total_cpu_shares / steps)
        average_num_containers.append(total_containers / steps)
        average_response_time.append(total_response_time / steps)
        adaptation_counts.append(adaptation_count)

    return costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations, average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time, adaptation_counts

if __name__ == '__main__':
    # env = AutoscaleEnv()
    num_episodes = 180
    costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations, average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time, adaptation_counts = q_learning(num_episodes)
    
    # Save the results to files
    #np.save('costs_per_episode.npy', costs_per_episode)
    #np.save('total_time_per_episode.npy', total_time_per_episode)
    #np.save('average_cost_per_episode.npy', average_cost_per_episode)
    #np.save('Rmax_violations.npy', Rmax_violations)
    #np.save('average_cpu_utilization.npy', average_cpu_utilization)
    #np.save('average_cpu_shares.npy', average_cpu_shares)
    #np.save('average_num_containers.npy', average_num_containers)
    #np.save('average_response_time.npy', average_response_time)
    #np.save('adaptation_counts.npy', adaptation_counts)

    # Define a function to plot each metric
    def plot_metric(iterations, metric, ylabel, title, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, metric, label=title)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Plot the results
    num_iterations = len(costs_per_episode)
    iterations = range(1, num_iterations + 1)
    
    plot_metric(iterations, costs_per_episode, 'Total Cost', 'Total Cost per Episode', '/app/plots/total_cost_per_episode.png')
    plot_metric(iterations, total_time_per_episode, 'Total Time', 'Total Time per Episode', '/app/plots/total_time_per_episode.png')
    plot_metric(iterations, average_cost_per_episode, 'Average Cost', 'Average Cost per Episode', '/app/plots/average_cost_per_episode.png')
    plot_metric(iterations, Rmax_violations, 'Rmax Violations (%)', 'Rmax Violations per Episode', '/app/plots/rmax_violations_per_episode.png')
    plot_metric(iterations, average_cpu_utilization, 'Average CPU Utilization (%)', 'Average CPU Utilization per Episode', '/app/plots/average_cpu_utilization_per_episode.png')
    plot_metric(iterations, average_cpu_shares, 'Average CPU Shares (%)', 'Average CPU Shares per Episode', '/app/plots/average_cpu_shares_per_episode.png')
    plot_metric(iterations, average_num_containers, 'Average Number of Containers', 'Average Number of Containers per Episode', '/app/plots/average_num_containers_per_episode.png')
    plot_metric(iterations, average_response_time, 'Average Response Time (ms)', 'Average Response Time per Episode', '/app/plots/average_response_time_per_episode.png')
    plot_metric(iterations, adaptation_counts, 'Adaptations (%)', 'Adaptations per Episode', '/app/plots/adaptations_per_episode.png')

    # Show the last plot
    plt.show()
