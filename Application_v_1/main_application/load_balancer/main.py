import numpy as np
import time
import prometheus_metrics
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from docker_api import DockerAPI
from costs import Costs
import docker
import pytz, os
import itertools

if not os.path.exists('/app/plots'):
    os.makedirs('/app/plots')

log_dir = '/app/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)   

timezone = pytz.timezone('Europe/Athens')
Rmax = 0.80 # 800 ms
seconds_for_next_episode = 60 # Determines the seconds for the next episode to begin.
alpha = 0.1
gamma = 0.99
epsilon_start = 1.0  # Starting value of epsilon
epsilon_end = 0.1    # Minimum value of epsilon
epsilon_decay = 0.98  # Decay factor per episode
cres = 0.01
wait_time = 15
url = 'http://prometheus:9090/api/v1/query'
service_name = 'mystack_application'
application_url = 'http://application:8501/train'
max_replicas = 10
min_replicas = 1
w_perf = 0.5 
w_res = 0.4
max_containers = 11

# Define the ranges for CPU utilization, number of running containers, and CPU shares
cpu_utilization_values = range(101)  # CPU utilization values from 0 to 100
k_range = range(1, max_containers)  # Number of running containers (1 to 10)
cpu_shares_values = [1024, 512, 256, 128]  # CPU shares (1024, 512, 256, 128)

# Generate all combinations of CPU utilization, K, and CPU shares
state_space = list(itertools.product(cpu_shares_values, cpu_utilization_values, k_range))
action_space = [-1, 0, 1, -512, 512]  # Actions: -1 (scale in), 0 (do nothing), 1 (scale out), -512 (decrease CPU shares), 512 (increase CPU shares)

# Initialize Q-table
# Q = np.zeros((len(state_space), len(action_space)))

iteration = 1

def reset_environment_to_initial_state():
    print("Log: Resetting the environemnt")
    scale_out(service_name=service_name, desired_replicas=1)
    set_cpu_shares(service_name, 2.0)

def transition(action):
    global was_transition_succefull
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
    retry_attempts = 5
    for attempt in range(retry_attempts):
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
            break  # Exit the loop if successful
        except KeyError as e:
            if 'NodeID' in str(e):
                print(f"Warning: 'NodeID' not found in service task on attempt {attempt + 1}. Retrying in 30 seconds...")
                time.sleep(30)
            else:
                print(f"Error: Unexpected KeyError: {e}")
                break
        except docker.errors.NotFound:
            print("Error: Service not found. Cannot increase CPU shares.")
            break
        except Exception as e:
            print(f"Error: An unexpected error occurred: {e}")
            break
    else:
        print("Error: Failed to set CPU shares after multiple attempts.")

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

def ensure_performance_penalty_has_data(performance_data):
    """
    Ensures that the performance penalty data is valid by checking if it is None or NaN.
    If the data is invalid, it repeatedly fetches new data until a valid performance penalty is obtained.

    Args:
        performance_data: The current performance penalty data to check. It can be None or NaN.

    Returns:
        The valid performance penalty data after it is successfully fetched and verified.
    
    Behavior:
        - If performance_data is None or NaN, it will print an error message and attempt to fetch valid data every 2 seconds.
        - The process repeats until a valid performance penalty (non-None, non-NaN) is retrieved.
    """
    while performance_data is None or np.isnan(performance_data):
        print("Error: Performance penalty is None or NaN. Retrying every 2 seconds ...")
        time.sleep(2)
        fetched_data = fetch_data()
        _, _, _, performance_data, _ = fetched_data
    return performance_data

def check_horizontal_or_vertical_scaling(action):
    """
    Determines whether the given action represents horizontal or vertical scaling.

    Args:
        action (int): The scaling action to check. Expected values are:
                      - 512 or -512 for horizontal scaling.
                      - 1 or -1 for vertical scaling.
                      - 0 for do nothing

    Returns:
        tuple: A tuple containing two boolean values:
               - The first indicates if the action represents horizontal scaling (True/False).
               - The second indicates if the action represents vertical scaling (True/False).
               If the action is neither, both values will be False.
    """
    if action == 512 or action == -512:
        return True, False
    elif action == 1 or action == -1:
        return False, True
    else:
        return False, False

def run_q_learning(num_episodes, w_perf, w_adp, w_res):
    
    episode = 1
    Rmax_violations = []
    costs_per_episode = []
    average_cpu_shares = []
    average_response_time = []
    average_num_containers = []
    total_time_per_episode = []
    avarage_vertical_scale = []
    average_cpu_utilization = []
    average_cost_per_episode = []
    avarage_horizontal_scale = []
    avarage_vertical_scale = []
    avarage_vertical_scale_per_episode = []
    avarage_horizontal_scale_per_episode = []
    avarage_containers_per_episode = []
    average_rmax_violations_per_episode = []
    average_cpu_utilization_per_episode = []
    
    total_actions = 0
    total_cpu_shares = 0
    total_response_time = 0
    vertical_scaling_count = 0
    horizontal_scaling_count = 0

    # Epsilon parameters
    epsilon = epsilon_start  # Start with the initial epsilon value

    print(f"\n\nGREETING: The run is expected to take approximately {(num_episodes * seconds_for_next_episode) / 60:.2f} minutes to complete. "
        "During this time, you may encounter some errors in the logs, but don't worry—the code is designed to handle them. "
        "\n\nThank you! Enjoy the process! 😊"
        "\n\nBest regards, "
        "\nVatistas Dimitrios"
        "\nSeptember 2024")
    print("\nLog: Training Starting ...")
    training_start_time = datetime.now()  # Start time of the entire training

    while episode <= num_episodes:
        print(f'Log: Episode: {episode}')
        app_state = state()
        total_cpu_utilization = 0
        total_Rmax_violations = 0
        total_containers = 0
        total_cost = 0
        total_reward = 0
        steps = 0
        Rmax_violation_count = 0
        next_state = app_state
        episode_start_time = datetime.now()  # Start time of the current episode

        while True:
            print("\n")
            current_state = next_state
            nearest_state = find_nearest_state(current_state, state_space)
            action = select_action(Q, nearest_state, epsilon)
            next_state = transition(action)
            
            if not was_transition_succefull:
                print('Log: No action because no transition was made.')
                action = 0
                
            fetched_data = fetch_data()  # Fetch data once per iteration
            
            _, _, _, performance_penalty, _ = fetched_data
            
            performance_penalty = ensure_performance_penalty_has_data(performance_penalty)
            print(f'Log: Perfomance Time: {performance_penalty}')
            total_response_time += performance_penalty
            print(f'Log: Total response time: {total_response_time}')
            
            a1 = 1 if action in [1, -1] else 0
            a2 = 1 if action in [-512, 512] else 0
            
            cost = Costs.overall_cost_function(w_adp, w_perf, w_res, next_state[2], next_state[1], next_state[0], action, a1, a2, Rmax, max_replicas, performance_penalty)
            
            total_cost += cost
            print(f'Log: Cost: {cost}, action: {action}')
            
            total_reward += cost
            total_cpu_utilization += current_state[1]
            total_cpu_shares += current_state[0]
            total_containers += current_state[2]
            steps += 1
            
            is_vertical_scale, is_horizontal_scale = check_horizontal_or_vertical_scaling(action)
            
            if is_horizontal_scale:
                horizontal_scaling_count += 1
                print(f'Horizontal scaling occurred: {horizontal_scaling_count}')

            if is_vertical_scale:
                vertical_scaling_count += 1
                print(f'Vertical scaling occurred: {vertical_scaling_count}')
            
            total_actions += 1
            print(f'Log: Response time: {performance_penalty:.2f}s')
            
            if performance_penalty > Rmax:
                Rmax_violation_count += 1
                total_Rmax_violations += 1
                print(f'Log: Rmax violation occured: Response time: {performance_penalty:.2f}s, Rmax: {Rmax}s, Total number of Violations: {Rmax_violation_count}')

            current_state_idx = state_space.index(nearest_state)
            next_state_idx = state_space.index(find_nearest_state(next_state, state_space))
            
            Q[current_state_idx, action_space.index(action)] = (
                (1 - alpha) * Q[current_state_idx, action_space.index(action)] +
                alpha * (cost + gamma * min(Q[next_state_idx, :]))
            )
            
            # Calculate ETA for the episode
            elapsed_time_episode = (datetime.now() - episode_start_time).total_seconds()
            average_time_per_step = elapsed_time_episode / steps if steps > 0 else 0
            remaining_steps = max(0, steps - 1)  # Assuming steps is an integer
            remaining_time_for_episode = remaining_steps * average_time_per_step
            eta_for_episode = datetime.now() + timedelta(seconds=remaining_time_for_episode)

            # Calculate ETA for all episodes
            elapsed_time_total = (datetime.now() - training_start_time).total_seconds()
            average_time_per_episode = elapsed_time_total / episode if episode > 0 else 0
            remaining_episodes = num_episodes - episode
            remaining_time_for_all_episodes = remaining_episodes * average_time_per_episode
            eta_for_all_episodes = datetime.now() + timedelta(seconds=remaining_time_for_all_episodes)

            athens_tz = pytz.timezone('Europe/Athens')
            eta_episode_athens = eta_for_episode.astimezone(athens_tz)
            eta_all_episodes_athens = eta_for_all_episodes.astimezone(athens_tz)

            print(f"Log: Episode: {episode}, ETA for current episode: {eta_episode_athens}, \nLog: ETA for all episodes: {eta_all_episodes_athens}")
            print(f'Log: Average response time of current episode: {total_response_time / steps:.2f}')
            print(f'Log: Average response time for all episodes so far: {total_response_time / steps:.2f}')
            print(f"Log: Action: {action}, Horizontal scaling: {is_horizontal_scale}, Vertical scaling: {is_vertical_scale}")
            
            if elapsed_time_episode > seconds_for_next_episode or steps >= 1000:
                break

        if steps > 0:
            costs_per_episode.append(total_cost / steps)
            total_time_per_episode.append(elapsed_time_episode / steps)
            average_cost_per_episode.append(total_reward / steps)
            Rmax_violations.append(Rmax_violation_count / steps)
            average_cpu_utilization.append(total_cpu_utilization / steps)
            average_cpu_shares.append(total_cpu_shares / steps)
            average_num_containers.append(total_containers / steps)
            average_response_time.append(total_response_time / steps)
            avarage_horizontal_scale.append(horizontal_scaling_count / steps)
            avarage_vertical_scale.append(vertical_scaling_count / steps)
            
            # Calculate metrics for cpu utilization
            average_cpu_utilization_for_episode = total_cpu_utilization / steps
            average_cpu_utilization_for_episode = min(average_cpu_utilization_for_episode, 100)
            average_cpu_utilization_per_episode.append(average_cpu_utilization_for_episode)
            
            # Calculate the Rmax violation percentage for the episode
            rmax_violation_percentage_for_episode = (Rmax_violation_count / steps) * 100
            average_rmax_violations_per_episode.append(rmax_violation_percentage_for_episode)
            
            avarage_horizontal_scale_for_episode = (horizontal_scaling_count / steps) * 100
            avarage_horizontal_scale_per_episode.append(avarage_horizontal_scale_for_episode)
            
            vertical_scaling_for_episode = (vertical_scaling_count / steps) * 100
            avarage_vertical_scale_per_episode.append(vertical_scaling_for_episode)
            
            # Calculate the avarage contaners
            avarage_containers_for_episode = (total_containers / steps)
            avarage_containers_per_episode.append(avarage_containers_for_episode)

        else:
            costs_per_episode.append(0)
            total_time_per_episode.append(0)
            average_cost_per_episode.append(0)
            Rmax_violations.append(0)
            average_cpu_utilization.append(0)
            average_cpu_shares.append(0)
            average_num_containers.append(0)
            average_response_time.append(0)
            avarage_horizontal_scale.append(0)
            avarage_vertical_scale.append(0)

        # Decay epsilon after each episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode += 1
        
    final_average_rmax_violations = sum(average_rmax_violations_per_episode) / len(average_rmax_violations_per_episode)
    final_average_cpu_utilization = sum(average_cpu_utilization_per_episode) / len(average_cpu_utilization_per_episode)
    final_avarage_containers = sum(avarage_containers_per_episode) / len(avarage_containers_per_episode)
    avarage_response_time = (total_response_time / total_actions)
    average_cpu_shares_new = (total_cpu_shares / total_actions)
    average_horizontal_scaling_final = sum(avarage_horizontal_scale_per_episode) / len(avarage_horizontal_scale_per_episode)
    avarage_vertical_scale_final = sum(avarage_vertical_scale_per_episode) / len(avarage_vertical_scale_per_episode)

    return (costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations,
            average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time,
            w_adp, w_perf, w_res, final_average_rmax_violations, final_average_cpu_utilization, final_avarage_containers, avarage_response_time, average_cpu_shares_new,
            average_horizontal_scaling_final, avarage_vertical_scale_final, avarage_horizontal_scale, avarage_vertical_scale, Q)
    
def run_baseline(num_episodes):
    episode = 1
    total_time_per_episode = []
    average_rmax_violations_per_episode = []
    average_cpu_utilization_per_episode = []
    average_response_time = []
    total_actions = 0

    print("Log: Baseline is now starting ...")
    training_start_time = datetime.now()  # Start time of the entire training

    while episode <= num_episodes:
        print(f'\nLog: Episode: {episode}')
        app_state = state()
        total_cpu_utilization = 0
        total_response_time = 0
        steps = 0
        Rmax_violation_count = 0
        next_state = app_state
        episode_start_time = datetime.now()  # Start time of the current episode

        while True:
            print("\n")
            current_state = next_state
                
            fetched_data = fetch_data()  # Fetch data once per iteration
            
            _, _, _, performance_penalty, _ = fetched_data
            
            performance_penalty = ensure_performance_penalty_has_data(performance_penalty)
            print(f'Log: Performance Time: {performance_penalty}')
            total_response_time += performance_penalty
            print(f'Log: Total response time: {total_response_time}')
                        
            total_cpu_utilization += current_state[1]
            steps += 1
            
            if performance_penalty > Rmax:
                Rmax_violation_count += 1
            
            time.sleep(10)
            
            # Calculate ETA for the episode
            elapsed_time_episode = (datetime.now() - episode_start_time).total_seconds()
            average_time_per_step = elapsed_time_episode / steps if steps > 0 else 0
            remaining_steps = max(0, steps - 1)  # Assuming steps is an integer
            remaining_time_for_episode = remaining_steps * average_time_per_step
            eta_for_episode = datetime.now() + timedelta(seconds=remaining_time_for_episode)

            # Calculate ETA for all episodes
            elapsed_time_total = (datetime.now() - training_start_time).total_seconds()
            average_time_per_episode = elapsed_time_total / episode if episode > 0 else 0
            remaining_episodes = num_episodes - episode
            remaining_time_for_all_episodes = remaining_episodes * average_time_per_episode
            eta_for_all_episodes = datetime.now() + timedelta(seconds=remaining_time_for_all_episodes)

            athens_tz = pytz.timezone('Europe/Athens')
            eta_episode_athens = eta_for_episode.astimezone(athens_tz)
            eta_all_episodes_athens = eta_for_all_episodes.astimezone(athens_tz)

            print(f"\nLog: Episode: {episode}, ETA for current episode: {eta_episode_athens}, \nLog: ETA for all episodes: {eta_all_episodes_athens}")
            print(f'Log: Average response time of current episode: {total_response_time / steps}')
            print(f'Log: Average response time for all episodes so far: {total_response_time / steps}')
            
            total_actions += 1
            
            if elapsed_time_episode > seconds_for_next_episode or steps >= 1000:
                break

        if steps > 0:
            # Calculate and store average CPU utilization for the episode
            average_cpu_utilization_for_episode = total_cpu_utilization / steps
            average_cpu_utilization_for_episode = min(average_cpu_utilization_for_episode, 100)  # Cap at 100%
            average_cpu_utilization_per_episode.append(average_cpu_utilization_for_episode)

            # Calculate and store Rmax violation percentage for the episode
            rmax_violation_percentage_for_episode = (Rmax_violation_count / steps) * 100
            average_rmax_violations_per_episode.append(rmax_violation_percentage_for_episode)

            # Store other episode-level metrics
            total_time_per_episode.append(elapsed_time_episode / steps)
            average_response_time.append(total_response_time / steps)
        else:
            total_time_per_episode.append(0)
            average_rmax_violations_per_episode.append(0)
            average_cpu_utilization_per_episode.append(0)
            average_response_time.append(0)

        episode += 1

    # Calculate the overall average CPU utilization across all episodes
    final_average_cpu_utilization = sum(average_cpu_utilization_per_episode) / len(average_cpu_utilization_per_episode)

    # Calculate the overall average Rmax violations across all episodes
    final_average_rmax_violations = sum(average_rmax_violations_per_episode) / len(average_rmax_violations_per_episode)

    # Calculate the overall average response time across all episodes
    final_average_response_time = sum(average_response_time) / len(average_response_time)

    return (total_time_per_episode, average_rmax_violations_per_episode, average_cpu_utilization_per_episode, average_response_time,
            final_average_rmax_violations, final_average_cpu_utilization, final_average_response_time)

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

def save_q_values(q, filename):
    np.save(filename, q)

def create_plots(run_number, iterations):
        plot_metric(iterations, costs_per_episode, 'Total Cost', 'Total Cost per Episode', f'/app/plots/total_cost_per_episode_{run_number}.png')
        plot_metric(iterations, total_time_per_episode, 'Total Time', 'Total Time per Episode', f'/app/plots/total_time_per_episode_{run_number}.png')
        plot_metric(iterations, average_cost_per_episode, 'Average Cost', 'Average Cost per Episode', f'/app/plots/average_cost_per_episode_{run_number}.png')
        plot_metric(iterations, Rmax_violations, 'Rmax Violations (%)', 'Rmax Violations per Episode', f'/app/plots/rmax_violations_per_episode_{run_number}.png')
        plot_metric(iterations, average_cpu_utilization, 'Average CPU Utilization (%)', 'Average CPU Utilization per Episode', f'/app/plots/average_cpu_utilization_per_episode_{run_number}.png')
        plot_metric(iterations, average_cpu_shares, 'Average CPU Shares (%)', 'Average CPU Shares per Episode', f'/app/plots/average_cpu_shares_per_episode_{run_number}.png')
        plot_metric(iterations, average_num_containers, 'Average Number of Containers', 'Average Number of Containers per Episode', f'/app/plots/average_num_containers_per_episode_{run_number}.png')
        plot_metric(iterations, average_response_time, 'Average Response Time (s)', 'Average Response Time per Episode', f'/app/plots/average_response_time_per_episode_{run_number}.png')
        plot_metric(iterations, avarage_horizontal_scale, 'Average Horizontal Scale', 'Average Horizontal Scale per Episode', f'/app/plots/average_horizontal_scale_per_episode_{run_number}.png')
        plot_metric(iterations, avarage_vertical_scale, 'Average Vertical Scale', 'Average Vertical Scale per Episode', f'/app/plots/average_response_time_per_episode_{run_number}.png')

def gather_learning_metrics_and_save(run_number, q, num_episodes, w_perf, w_res, w_adp, Rmax, rmax_violations_percantage, 
                                     cpu_utilization_percentage, average_cpu_shares_new, containers_percentage, 
                                     avarage_response_time_new, average_horizontal_scaling_final = 0, avarage_vertical_scale_final = 0):
    q_learning_log_path = f'/logs/q-learning-final-log_{run_number}.txt'
    q_learning_values_path = f'/logs/q-values_{run_number}.npy'
    q_learning_statistics = (
            f"Q-learning Final Episode Statistics:\n"
            f"Estimated Running Time: {(num_episodes * seconds_for_next_episode) / 60:.2f} in minutes\n"
            f"Wperf = {w_perf}, Wres = {w_res}, Wadp = {w_adp}, Rmax = {Rmax}\n"
            f"Rmax Violations: {rmax_violations_percantage:.2f}%\n"
            f"Average CPU Utilization: {cpu_utilization_percentage:.2f}%\n"
            f"Average CPU Shares: {average_cpu_shares_new:.2f}\n"
            f"Average Number of Containers: {containers_percentage:.2f}\n"
            f"Average Response Time: {avarage_response_time_new:.2f} s\n"
            f"Average Vertical Scale: {avarage_vertical_scale_final:.2f} %\n"
            f"Average Horizontal Scale: {average_horizontal_scaling_final:.2f} %\n"
        )
    save_final_statistics(q_learning_statistics, q_learning_log_path)
    save_q_values(q, q_learning_values_path)
    print(q_learning_statistics)
    
if __name__ == '__main__':
    
    reset_environment_to_initial_state()
    
    num_episodes = 1

    baseline = True
    
    run_first = True
    
    run_second = True    
  
    run_third = True
    
    if run_first:
        
        # Initialize Q-table
        Q = np.zeros((len(state_space), len(action_space)))
        q_learning_metrics = run_q_learning(num_episodes, w_perf=0.90, w_adp=0.01, w_res=0.09)
        
        (costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations,
        average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time,
        w_adp, w_perf, w_res, rmax_violations_percantage, cpu_utilization_percentage, containers_percentage, avarage_response_time_new, average_cpu_shares_new,
        average_horizontal_scaling_final, avarage_vertical_scale_final, avarage_horizontal_scale, avarage_vertical_scale, q) = q_learning_metrics
        
        num_iterations = len(costs_per_episode)
        iterations = range(1, num_iterations + 1)
        running_time = num_episodes * seconds_for_next_episode / 60
        
        create_plots(run_number="First", iterations=iterations)
        gather_learning_metrics_and_save("First", q, running_time, w_perf, w_res, w_adp, Rmax, rmax_violations_percantage, cpu_utilization_percentage, average_cpu_shares_new, containers_percentage, avarage_response_time_new)

    reset_environment_to_initial_state()
    
    if run_second:
        # Initialize Q-table
        Q = np.zeros((len(state_space), len(action_space)))
        q_learning_metrics = run_q_learning(num_episodes, w_perf=0.09, w_adp=0.01, w_res=0.90)

        (costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations,
        average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time,
        w_adp, w_perf, w_res, rmax_violations_percantage, cpu_utilization_percentage, containers_percentage, avarage_response_time_new, average_cpu_shares_new,
        average_horizontal_scaling_final, avarage_vertical_scale_final, avarage_horizontal_scale, avarage_vertical_scale, q) = q_learning_metrics

        num_iterations = len(costs_per_episode)
        iterations = range(1, num_iterations + 1)

        create_plots(run_number="Second", iterations=iterations)
        gather_learning_metrics_and_save("Second", q, num_episodes, w_perf, w_res, w_adp, Rmax, rmax_violations_percantage, cpu_utilization_percentage, average_cpu_shares_new, containers_percentage, avarage_response_time_new)

    reset_environment_to_initial_state()

    if run_third:
        # Initialize Q-table
        Q = np.zeros((len(state_space), len(action_space)))
        q_learning_metrics = run_q_learning(num_episodes, w_perf=0.33, w_adp=0.33, w_res=0.33)

        (costs_per_episode, total_time_per_episode, average_cost_per_episode, Rmax_violations,
        average_cpu_utilization, average_cpu_shares, average_num_containers, average_response_time,
        w_adp, w_perf, w_res, rmax_violations_percantage, cpu_utilization_percentage, containers_percentage, avarage_response_time_new, average_cpu_shares_new,
        average_horizontal_scaling_final, avarage_vertical_scale_final, avarage_horizontal_scale, avarage_vertical_scale, q) = q_learning_metrics

        num_iterations = len(costs_per_episode)
        iterations = range(1, num_iterations + 1)

        create_plots(run_number="Third", iterations=iterations)
        gather_learning_metrics_and_save("Third", q, num_episodes, w_perf, w_res, w_adp, Rmax, rmax_violations_percantage, cpu_utilization_percentage, average_cpu_shares_new, containers_percentage, avarage_response_time_new)

    reset_environment_to_initial_state()
    
    if baseline:
        # Run baseline scenario (without Q-learning) and gather metrics
        baseline_metrics = run_baseline(num_episodes)
        
        # Extract metrics
        total_time_per_episode, Rmax_violations, average_cpu_utilization, average_response_time, rmax_violations_percantage, cpu_utilization_percentage, avarage_response_time= baseline_metrics
        
        num_iterations = len(costs_per_episode)
        iterations = range(1, num_iterations + 1)
        
        # Plot and save baseline results
        plot_metric(iterations, total_time_per_episode, 'Total Time', 'Total Time per Episode (Baseline)', '/app/plots/total_time_per_episode_baseline.png')
        plot_metric(iterations, Rmax_violations, 'Rmax Violations (%)', 'Rmax Violations per Episode (Baseline)', '/app/plots/rmax_violations_per_episode_baseline.png')
        plot_metric(iterations, average_cpu_utilization, 'Average CPU Utilization (%)', 'Average CPU Utilization per Episode (Baseline)', '/app/plots/average_cpu_utilization_per_episode_baseline.png')
        plot_metric(iterations, average_num_containers, 'Average Number of Containers', 'Average Number of Containers per Episode (Baseline)', '/app/plots/average_num_containers_per_episode_baseline.png')
        plot_metric(iterations, average_response_time, 'Average Response Time (ms)', 'Average Response Time per Episode (Baseline)', '/app/plots/average_response_time_per_episode_baseline.png')

        # Prepare final episode statistics for baseline
        baseline_statistics = (
            f"Baseline Final Episode Statistics:\n"
            f"Rmax Violations: {Rmax_violations[-1] * 100 / num_episodes:.2f}%\n"
            f"Average CPU Utilization: {average_cpu_utilization[-1]:.2f}%\n"
            f"Average Number of Containers: {average_num_containers[-1]:.2f}\n"
            f"Average Response Time: {average_response_time[-1]:.2f} ms\n"
            
        )

        # Save final episode statistics for baseline to a log file
        baseline_log_path = '/logs/baseline-final-log.txt'
        save_final_statistics(baseline_statistics, baseline_log_path)
        print(baseline_statistics)

        # Show the last plot (optional)
        plt.show()