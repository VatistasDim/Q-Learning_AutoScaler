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

log_dir = '/app/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)   

# Constants
timezone = pytz.timezone('Europe/Athens')
wperf = 0.09
wres = 0.90
wadp = 0.01
Rmax = 0.5 #This is in ms
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

def reset_environment_to_initial_state():
    print("Log: Resetting the environemnt")
    scale_out(service_name=service_name, desired_replicas=1)
    set_cpu_shares(service_name, 2.0)

def transition(action):
    global was_transition_succefull  # Declare the global variable
    running_containers = get_current_replica_count(service_prefix=service_name)
    current_cpu_shares = get_current_cpu_shares(service_name)
    print(f"Log: Current CPU shares from transition: {current_cpu_shares}")

    if action == -1:  # Scale in (decrease containers)
        print("Log: Decrease Container by 1")
        was_transition_succefull = scale_in(service_name=service_name, scale_out_factor=1)
    elif action == 1:  # Scale out (increase containers)
        print("Log: Increase Container by 1")
        desired_replicas = running_containers + 1
        was_transition_succefull = scale_out(service_name=service_name, desired_replicas=desired_replicas)
    elif action == -512:  # Decrease CPU shares
        print("Log: Decrease CPU shares")
        was_transition_succefull = decrease_cpu_share_step(current_cpu_share=current_cpu_shares)
    elif action == 512:  # Increase CPU shares
        print("Log: Increase CPU shares")
        was_transition_succefull = increase_cpu_share_step(current_cpu_share=current_cpu_shares)
    elif action == 0:
        print("Log: No Action")
        time.sleep(15)
        was_transition_succefull = True

    c, u, k = state()
    new_cpu_shares = get_current_cpu_shares(service_name)
    print(f"Log: New CPU shares after action: {new_cpu_shares}")
    return (c, u, k)


def increase_cpu_share_step(current_cpu_share):
    print(f'Log: increase_cpu_share_step --> current_cpu_share:{current_cpu_share}')
    if current_cpu_share == 1:
        set_cpu_shares(service_name, 2.0)
        return True
    elif current_cpu_share == 2:
        print(f"Log: No Increase, already at max cpu shares")
        return False

def decrease_cpu_share_step(current_cpu_share):
    print(f'Log: decrease_cpu_share_step --> current_cpu_share:{current_cpu_share}')
    if current_cpu_share == 2:
        set_cpu_shares(service_name, 1.0)
        return True
    elif current_cpu_share == 1:
        print("Log: No decrease in cpu shares, already at lowest cpu shares")
        return False

def select_action(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(action_space)
    else:
        state_idx = state_space.index(state)
        return action_space[np.argmin(Q[state_idx])]

def fetch_data():
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            cpu_percent, ram_percent, time_up, response_time, cpu_shares = prometheus_metrics.start_metrics_service(url=url)
            if time_up != '0':
                cpu_percent = int(float(cpu_percent))
                ram_percent = int(float(ram_percent))
                time_up = int(float(time_up))
                response_time = float(response_time)
                cpu_shares = calculate_cpu_shares(get_current_cpu_shares(service_name))
                if None in (cpu_percent, ram_percent, time_up, response_time, cpu_shares):
                    continue
                return cpu_percent, ram_percent, time_up, response_time, cpu_shares
        except Exception as e:
            print(f"Error: An error occurred during service metrics retrieval (Attempt {attempt + 1}/{max_attempts}):", e)
            if attempt < max_attempts - 1:
                time.sleep(60)
    print("Failed to retrieve metrics after multiple attempts.")
    return None, None, None, None, None

def calculate_cpu_shares(cpu_fraction):
    if cpu_fraction == 2.0:
        return 2 # 2 CPU
    elif cpu_fraction == 1.0:
        return 1 # 1 CPU
    elif cpu_fraction == 0.5:
        return 0.5 # 0.5 CPU
    elif cpu_fraction == 0.25:
        return 0.25 # 0.25 CPU
    elif cpu_fraction == 0.125:
        return 0.125 # 0.125 CPU
    else:
        return 2 # Default to 2 CPU

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

def get_node_resources(node_id):
    client = docker.from_env()
    try:
        node = client.nodes.get(node_id)
        node_info = node.attrs
        resources = node_info['Description']['Resources']
        return resources['NanoCPUs'], resources['MemoryBytes']
    except docker.errors.NotFound:
        return None, None

def set_cpu_shares(service_name, cpu_shares):
    client = docker.from_env()
    try:
        print("Log: Setting CPU shares")
        service = client.services.get(service_name)
        service_tasks = service.tasks()
        
        if not service_tasks:
            print("Error: No tasks found for the service")
            return
        
        node_id = service_tasks[0]['NodeID']
        node_nano_cpus, node_memory_bytes = get_node_resources(node_id)
        
        if node_nano_cpus is None or node_memory_bytes is None:
            print("Error: Could not get node resources")
            return
        
        print(f'Log: Node Nano CPUs: {node_nano_cpus}, Node Memory Bytes: {node_memory_bytes}')
        
        resources = service.attrs['Spec']['TaskTemplate']['Resources']
        if 'Limits' not in resources:
            resources['Limits'] = {}
        
        current_cpu_shares = resources['Limits'].get('NanoCPUs', 0)
        print(f'Log: Current CPU Shares: {current_cpu_shares} NanoCPUs')
        
        desired_cpu_shares_nano = int(cpu_shares * 1_000_000_000)
        print(f'Log: Desired CPU Shares: {desired_cpu_shares_nano} NanoCPUs')
        
        if desired_cpu_shares_nano > node_nano_cpus:
            print("Error: Not enough CPU resources available")
            return
        
        resources['Limits']['NanoCPUs'] = desired_cpu_shares_nano
        service.update(resources=resources)
        print(f"Log: CPU shares set to {desired_cpu_shares_nano} NanoCPUs for service {service_name}")
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
    if desired_replicas <= max_replicas:
        service = client.services.get(service_name)
        service.scale(desired_replicas)
        print(f"Log: Service '{service_name}' scaled to {desired_replicas} replicas.")
        time.sleep(15)
        return True
    else:
        print("Log: Maximum containers reached, transition not possible.")
        return False

def scale_in(service_name, scale_out_factor):
    client = docker.from_env()
    service = client.services.get(service_name)
    current_replicas = get_current_replica_count(service_name)
    if current_replicas != 1:
        desired_replicas = current_replicas - scale_out_factor
        service.scale(desired_replicas)
        print(f"Log: Service '{service_name}' scaled to {desired_replicas} replicas.")
        time.sleep(15)
        return True
    else:
        print(f'Log: Minimum containers reached, transtion is not possible.')
        return False

def state():
    docker_client = DockerAPI(service_name)
    cpu_value, _, _, _, c_cpu_shares = fetch_data()
    c_cpu_shares = get_current_cpu_shares(service_name)
    u_cpu_utilization = cpu_value
    print(f'Log: CPU Shares Real: {c_cpu_shares}')
    k_running_containers = get_current_replica_count(service_prefix=service_name)
    return c_cpu_shares, u_cpu_utilization, k_running_containers

def find_nearest_state(state, state_space):
    distances = [sum(abs(np.array(state) - np.array(s))) for s in state_space]
    nearest_index = np.argmin(distances)
    return state_space[nearest_index]

def run_q_learning(num_episodes):
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
    
    print("Log: Training Starting ... \n")

    while episode <= num_episodes:
        print(f'\n Log: Epidose: {episode}')
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
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print("\n")
            current_state = next_state
            nearest_state = find_nearest_state(current_state, state_space)
            action = select_action(Q, nearest_state, epsilon)
            next_state = transition(action)
            
            fetched_data = fetch_data()  # Fetch data once per iteration
            if fetched_data[3] is None:
                print("Error: Performance penalty is None. Skipping calculation.")
                performance_penalty = 0
                break
            else:
                _, _, _, performance_penalty, _ = fetched_data

            resource_cost = cres * float(next_state[2])
            a1 = 1 if action in [1, -1] else 0
            a2 = 1 if action in [-512, 512] else 0
            cost = Costs.overall_cost_function(wadp, w_perf, w_res, next_state[2], next_state[1], next_state[0], action, a1, a2, Rmax, max_replicas, performance_penalty)
            if was_transition_succefull:
                print('Info: Cost will be affected by 10 because no transition was made.')
                cost = 100
            total_cost += cost
            print(f'Log: Cost: {cost}, action: {action}')
            print(f'Total Cost: {total_cost}')
            total_reward += total_cost
            total_cpu_utilization += current_state[1]
            total_cpu_shares += current_state[0]
            total_containers += current_state[2]
            total_response_time += fetched_data[3]
            
            steps += 1
            
            if action != 0:
                adaptation_count += 1
            
            if performance_penalty > Rmax:
                Rmax_violation_count += 1

            current_state_idx = state_space.index(nearest_state)
            next_state_idx = state_space.index(find_nearest_state(next_state, state_space))
            
            Q[current_state_idx, action_space.index(action)] = (
                (1 - alpha) * Q[current_state_idx, action_space.index(action)] +
                alpha * (cost + gamma * min(Q[next_state_idx, :]))
            )
            
            total_time += elapsed_time
            # Calculate ETA
            remaining_episodes = num_episodes - episode
            average_time_per_episode = total_time / episode if episode > 0 else 0
            remaining_time = remaining_episodes * average_time_per_episode
            eta = datetime.now() + timedelta(seconds=remaining_time)
            athens_tz = pytz.timezone('Europe/Athens')
            eta_athens = eta.astimezone(athens_tz)

            print(f"Log: Episode: {episode}, ETA: {eta_athens}")

            if elapsed_time > 60:
                break  # Breaking if elapsed time is more than 1 minute

        episode += 1
        costs_per_episode.append(total_cost / steps)
        total_time_per_episode.append(total_time / steps)
        average_cost_per_episode.append(total_reward / steps)
        Rmax_violations.append(Rmax_violation_count)
        average_cpu_utilization.append(total_cpu_utilization / steps)
        average_cpu_shares.append(total_cpu_shares / steps)
        average_num_containers.append(total_containers / steps)
        average_response_time.append(total_response_time / steps)
        adaptation_counts.append(adaptation_count / steps)

    return (costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations,
            average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time, adaptation_counts)

def run_baseline(num_episodes):
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
            
            next_state = transition(None)  # No Q-learning action selection
            
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

            if datetime.now() - start_time > timedelta(minutes=1):
                break

        episode += 1
        costs_per_episode.append(total_cost / steps)
        total_time_per_episode.append(total_time / steps)
        average_cost_per_episode.append(total_reward / steps)
        Rmax_violations.append(Rmax_violation_count)
        average_cpu_utilization.append(total_cpu_utilization / steps)
        average_cpu_shares.append(total_cpu_shares / steps)
        average_num_containers.append(total_containers / steps)
        average_response_time.append(total_response_time / steps)
        adaptation_counts.append(adaptation_count / steps)

    return costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations, average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time, adaptation_counts

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

def save_final_statistics(statistics, filename):
    with open(filename, 'w') as log_file:
        log_file.write(statistics)

if __name__ == '__main__':
    
    num_episodes = 3

    baseline = False
    
    # Run Q-learning and gather metrics
    q_learning_metrics = run_q_learning(num_episodes)
    
    # Extract metrics
    (costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations,
    average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time, adaptation_counts) = q_learning_metrics
    
    # Plot and save Q-learning results
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

    # Prepare final episode statistics
    q_learning_statistics = (
        f"Q-learning Final Episode Statistics:\n"
        f"Estimated Running Time: {num_episodes} in minutes\n"
        f"Wperf = {wperf}, Wres = {wres}, Wadp = {wadp}, Rmax = {Rmax}\n"
        f"Rmax Violations: {Rmax_violations[-1] * 100 / num_episodes:.2f}%\n"
        f"Average CPU Utilization: {average_cpu_utilization[-1]:.2f}%\n"
        f"Average CPU Shares: {average_cpu_shares[-1]:.2f}%\n"
        f"Average Number of Containers: {average_num_containers[-1]:.2f}\n"
        f"Average Response Time: {average_response_time[-1]:.2f} ms\n"
        f"Adaptations: {adaptation_counts[-1] * 100 / num_episodes:.2f}%\n"
    )

    # Save Q-learning final episode statistics to a log file
    q_learning_log_path = '/logs/q-learning-final-log.txt'
    save_final_statistics(q_learning_statistics, q_learning_log_path)
    # Print and save final episode statistics
    print(q_learning_statistics)

    # Reset the environment to its initial state
    reset_environment_to_initial_state()
    
    if baseline:
        # Run baseline scenario (without Q-learning) and gather metrics
        baseline_metrics = run_baseline(num_episodes)
        
        # Extract metrics
        costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations, average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time, adaptation_counts = baseline_metrics

        # Plot and save baseline results
        plot_metric(iterations, costs_per_episode, 'Total Cost', 'Total Cost per Episode (Baseline)', '/app/plots/total_cost_per_episode_baseline.png')
        plot_metric(iterations, total_time_per_episode, 'Total Time', 'Total Time per Episode (Baseline)', '/app/plots/total_time_per_episode_baseline.png')
        plot_metric(iterations, average_cost_per_episode, 'Average Cost', 'Average Cost per Episode (Baseline)', '/app/plots/average_cost_per_episode_baseline.png')
        plot_metric(iterations, Rmax_violations, 'Rmax Violations (%)', 'Rmax Violations per Episode (Baseline)', '/app/plots/rmax_violations_per_episode_baseline.png')
        plot_metric(iterations, average_cpu_utilization, 'Average CPU Utilization (%)', 'Average CPU Utilization per Episode (Baseline)', '/app/plots/average_cpu_utilization_per_episode_baseline.png')
        plot_metric(iterations, average_cpu_shares, 'Average CPU Shares (%)', 'Average CPU Shares per Episode (Baseline)', '/app/plots/average_cpu_shares_per_episode_baseline.png')
        plot_metric(iterations, average_num_containers, 'Average Number of Containers', 'Average Number of Containers per Episode (Baseline)', '/app/plots/average_num_containers_per_episode_baseline.png')
        plot_metric(iterations, average_response_time, 'Average Response Time (ms)', 'Average Response Time per Episode (Baseline)', '/app/plots/average_response_time_per_episode_baseline.png')
        plot_metric(iterations, adaptation_counts, 'Adaptations (%)', 'Adaptations per Episode (Baseline)', '/app/plots/adaptations_per_episode_baseline.png')

        # Prepare final episode statistics for baseline
        baseline_statistics = (
            f"Baseline Final Episode Statistics:\n"
            f"Rmax Violations: {Rmax_violations[-1] * 100 / num_episodes:.2f}%\n"
            f"Average CPU Utilization: {average_cpu_utilization[-1]:.2f}%\n"
            f"Average CPU Shares: {average_cpu_shares[-1]:.2f}%\n"
            f"Average Number of Containers: {average_num_containers[-1]:.2f}\n"
            f"Average Response Time: {average_response_time[-1]:.2f} ms\n"
            f"Adaptations: {adaptation_counts[-1] * 100 / num_episodes:.2f}%\n"
        )

        # Save final episode statistics for baseline to a log file
        baseline_log_path = '/logs/baseline-final-log.txt'
        save_final_statistics(baseline_statistics, baseline_log_path)
        print(baseline_statistics)

        # Show the last plot (optional)
        plt.show()
