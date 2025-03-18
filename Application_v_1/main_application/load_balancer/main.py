import numpy as np # type: ignore
import prometheus_metrics
import matplotlib.pyplot as plt # type: ignore
from datetime import datetime, timedelta
from docker_api import DockerAPI
from costs import Costs
from generate_q_learning_weights import check_and_delete_file, create_file_with_random_weights
from settings import load_settings
import pytz, os, itertools, docker, time # type: ignore

if not os.path.exists('/app/plots'):
    os.makedirs('/app/plots')

log_dir = '/app/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)   
client = docker.DockerClient(base_url='unix://var/run/docker.sock')

settings_file = 'ApplicationSettings/ApplicationSettings.txt'
settings = load_settings(settings_file)
timezone = pytz.timezone('Europe/Athens')
Rmax = settings.get('Rmax', 0.80)
total_episodes = settings.get('total_episodes', 100)
seconds_for_next_episode = settings.get('seconds_for_next_episode', 60)
alpha = settings.get('alpha', 0.1)
gamma = settings.get('gamma', 0.99)
epsilon_start = settings.get('epsilon_start', 1.0)
epsilon_end = settings.get('epsilon_end', 0.1)
epsilon_decay = settings.get('epsilon_decay', 0.98)
cres = 0.01
wait_time = settings.get('wait_time', 15)
baseline = settings.get('baseline', False)
url = settings.get('url', 'http://prometheus:9090/api/v1/query')
service_name = settings.get('service_name', 'mystack_application')
max_replicas = settings.get('max_replicas', 10)
min_replicas = settings.get('min_replicas', 1)
max_containers = settings.get('max_containers', 11)

print(f'Rmax: {Rmax}')
print(f'Total Episodes: {total_episodes}')
print(f'seconds_for_next_episode: {seconds_for_next_episode}')
print(f'alpha: {alpha}')
print(f'gamma: {gamma}')
print(f'epsilon_start: {epsilon_start}')
print(f'epsilon_end: {epsilon_end}')
print(f'epsilon_decay: {epsilon_decay}')
print(f'wait_time: {wait_time}')
print(f'baseline: {baseline}')
print(f'url: {url}')
print(f'service_name: {service_name}')
print(f'max_replicas: {max_replicas}')
print(f'min_replicas: {min_replicas}')
print(f'max_containers: {max_containers}')


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

    scaling_type = None

    if action == -1:  # Scale in (decrease containers)
        print("Log: Decrease Container by 1")
        was_transition_succefull = scale_in(service_name=service_name, scale_out_factor=1)
        scaling_type = "horizontal"
    elif action == 1:  # Scale out (increase containers)
        print("Log: Increase Container by 1")
        desired_replicas = running_containers + 1
        was_transition_succefull = scale_out(service_name=service_name, desired_replicas=desired_replicas)
        scaling_type = "horizontal"
    elif action == -512:  # Decrease CPU shares
        print("Log: Decrease CPU shares")
        was_transition_succefull = decrease_cpu_share_step(current_cpu_share=current_cpu_shares)
        scaling_type = "vertical"
    elif action == 512:  # Increase CPU shares
        print("Log: Increase CPU shares")
        was_transition_succefull = increase_cpu_share_step(current_cpu_share=current_cpu_shares)
        scaling_type = "vertical"
    elif action == 0:
        print("Log: No Action")
        time.sleep(15)
        was_transition_succefull = True

    c, u, k = state()
    new_cpu_shares = get_current_cpu_shares(service_name)
    print(f"Log: New CPU shares after action: {new_cpu_shares}")
    
    return (c, u, k), scaling_type  # Return the scaling type

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

def print_training_start_message(num_episodes):
    print(f"Starting Q-learning training for {num_episodes} episodes...")
    
def log_episode_progress(episode, episode_start_time, training_start_time, steps, metrics):
    elapsed_time = (datetime.now() - training_start_time).total_seconds()
    episode_duration = (datetime.now() - episode_start_time).total_seconds()
    print(f"Episode {episode} completed in {episode_duration:.2f} seconds.")
    print(f"Total training time elapsed: {elapsed_time:.2f} seconds.")

    if episode > 1 and 'average_cost_per_episode' in metrics:
        recent_costs = metrics['average_cost_per_episode'][-5:] if len(metrics['average_cost_per_episode']) >= 5 else metrics['average_cost_per_episode']
        if len(recent_costs) > 1 and recent_costs[-1] < np.mean(recent_costs[:-1]):
            print("Log: Algorithm appears to be learning, as costs are decreasing.")
        else:
            print("Log: Algorithm may not be learning effectively, as costs are not decreasing.")

def run_q_learning(num_episodes, w_perf, w_adp, w_res):
    episode = 1
    metrics = initialize_metrics()
    total_actions, total_cpu_shares, total_response_time = 0, 0, 0
    vertical_scaling_count, horizontal_scaling_count = 0, 0
    total_scaling_actions = 0
    total_cost_per_episode = []
    total_cpu_utilization_per_episode = []
    epsilon = epsilon_start
    seconds_for_next_episode = 60  # Ensure each episode lasts 60 seconds

    print_training_start_message(num_episodes)
    training_start_time = datetime.now()

    while episode <= num_episodes:
        episode_start_time = datetime.now()
        print(f'Log: Episode: {episode}')
        episode_metrics = initialize_episode_metrics()
        next_state, steps = state(), 0
        episode_total_cost = 0
        episode_cpu_utilization = 0

        while not should_terminate_episode(episode_start_time):
            steps += 1
            current_state = next_state
            nearest_state = find_nearest_state(current_state, state_space)
            action = select_action(Q, nearest_state, epsilon)
            next_state, scaling_type = transition(action)

            if not was_transition_succefull:
                print('Log: No action because no transition was made.')
                action = 0

            if scaling_type == "horizontal":
                horizontal_scaling_count += 1
            elif scaling_type == "vertical":
                vertical_scaling_count += 1

            total_scaling_actions += 1
            
            fetched_data = fetch_data()
            performance_penalty = ensure_performance_penalty_has_data(fetched_data[3])
            cost = compute_cost(w_adp, w_perf, w_res, next_state, action, performance_penalty)

            episode_total_cost += cost
            episode_cpu_utilization += fetched_data[1]
            
            update_metrics(metrics, episode_metrics, action, performance_penalty, cost, current_state)
            update_q_table(Q, nearest_state, next_state, action, cost)
            log_episode_progress(episode, episode_start_time, training_start_time, steps, metrics)

        finalize_episode_metrics(metrics, episode_metrics, steps)
        total_cost_per_episode.append(episode_total_cost)
        total_cpu_utilization_per_episode.append(episode_cpu_utilization / steps)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode += 1

        if total_scaling_actions > 0:
            horizontal_scaling_percentage = (horizontal_scaling_count / total_scaling_actions) * 100
            vertical_scaling_percentage = (vertical_scaling_count / total_scaling_actions) * 100
        else:
            horizontal_scaling_percentage = 0
            vertical_scaling_percentage = 0

        final_metrics = compute_final_metrics(metrics, total_response_time, total_actions, total_cpu_shares,
                                          total_cost_per_episode, total_cpu_utilization_per_episode,
                                          horizontal_scaling_percentage, vertical_scaling_percentage)
        return (total_cost_per_episode,
                    np.mean(metrics['average_containers_per_episode']),
                    final_metrics['average_response_time'],
                    final_metrics['average_horizontal_scaling_final'],
                    final_metrics['average_vertical_scale_final'],
                    final_metrics['average_cost_per_episode'],
                    final_metrics['average_cpu_utilization_per_episode'],
                    w_adp, w_perf, w_res, Q,
                    final_metrics['horizontal_scaling_percentage'],
                    final_metrics['vertical_scaling_percentage'])

def initialize_metrics():
    return {
        'costs_per_episode': [],
        'total_time_per_episode': [],
        'average_cost_per_episode': [],
        'Rmax_violations': [],
        'average_cpu_utilization': [],
        'average_cpu_shares': [],
        'average_num_containers': [],
        'average_response_time': [],
        'average_horizontal_scale': [],
        'average_vertical_scale': [],
        'average_horizontal_scale_per_episode': [],
        'average_vertical_scale_per_episode': [],
        'average_containers_per_episode': [],
        'average_rmax_violations_per_episode': [],
        'average_cpu_utilization_per_episode': []
    }

def initialize_episode_metrics():
    return {
        'total_cpu_utilization': 0,
        'horizontal_scale_percentage' : 0,
        'vertical_scale_percentage' : 0,
        'total_Rmax_violations': 0,
        'total_containers': 0,
        'total_cost': 0,
        'total_reward': 0,
        'Rmax_violation_count': 0
    }

def should_terminate_episode(start_time):
    return (datetime.now() - start_time).total_seconds() >= 60

def compute_cost(w_adp, w_perf, w_res, next_state, action, performance_penalty):
    a1, a2 = int(action in [1, -1]), int(action in [-512, 512])
    return Costs.overall_cost_function(w_adp, w_perf, w_res, next_state[2], next_state[1], next_state[0], action, a1, a2, Rmax, max_replicas, performance_penalty)

def update_metrics(metrics, episode_metrics, action, performance_penalty, cost, current_state):
    episode_metrics['total_reward'] += cost
    episode_metrics['total_cpu_utilization'] += current_state[1]
    episode_metrics['total_containers'] += current_state[2]
    episode_metrics['total_cost'] += cost

    if performance_penalty > Rmax:
        episode_metrics['Rmax_violation_count'] += 1
        episode_metrics['total_Rmax_violations'] += 1

def update_q_table(Q, nearest_state, next_state, action, cost):
    current_state_idx = state_space.index(nearest_state)
    next_state_idx = state_space.index(find_nearest_state(next_state, state_space))
    Q[current_state_idx, action_space.index(action)] = (
        (1 - alpha) * Q[current_state_idx, action_space.index(action)] +
        alpha * (cost + gamma * min(Q[next_state_idx, :]))
    )

def finalize_episode_metrics(metrics, episode_metrics, steps):
    if steps > 0:
        metrics['costs_per_episode'].append(episode_metrics['total_cost'] / steps)
        metrics['average_cost_per_episode'].append(episode_metrics['total_reward'] / steps)
        metrics['Rmax_violations'].append(episode_metrics['Rmax_violation_count'] / steps)
        metrics['average_cpu_utilization'].append(episode_metrics['total_cpu_utilization'] / steps)
        metrics['average_containers_per_episode'].append(episode_metrics['total_containers'] / steps)
    else:
        for key in metrics:
            metrics[key].append(0)

def compute_final_metrics(metrics, total_response_time, total_actions, total_cpu_shares, 
                          total_cost_per_episode, total_cpu_utilization_per_episode,
                          horizontal_scaling_percentage, vertical_scaling_percentage):
    return {
        'final_average_rmax_violations': np.mean(metrics['average_rmax_violations_per_episode']),
        'final_average_cpu_utilization': np.mean(metrics['average_cpu_utilization_per_episode']),
        'final_average_containers': np.mean(metrics['average_containers_per_episode']),
        'average_response_time': (total_response_time / total_actions) if total_actions > 0 else 0,
        'average_cpu_shares_new': (total_cpu_shares / total_actions) if total_actions > 0 else 0,
        'average_cost_per_episode': np.mean(total_cost_per_episode) if total_cost_per_episode else 0,
        'average_cpu_utilization_per_episode': np.mean(total_cpu_utilization_per_episode) if total_cpu_utilization_per_episode else 0,
        'average_horizontal_scaling_final': np.mean(metrics['average_horizontal_scale_per_episode']),
        'average_vertical_scale_final': np.mean(metrics['average_vertical_scale_per_episode']),
        'horizontal_scaling_percentage': horizontal_scaling_percentage,
        'vertical_scaling_percentage': vertical_scaling_percentage
    }
    
# def format_output(metrics, final_metrics, w_adp, w_perf, w_res, Q):
    return (
        metrics['costs_per_episode'],  
        final_metrics['final_average_containers'],  
        final_metrics['average_response_time'],  
        final_metrics['average_horizontal_scaling_final'],  
        final_metrics['average_vertical_scale_final'],  
        final_metrics['average_cost_per_episode'],
        final_metrics['final_average_cpu_utilization'],
        w_adp, w_perf, w_res, Q
    )
    
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

def plot_metric(x, y, ylabel, title, filename):
    """Generic function to plot a metric."""
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def save_final_statistics(statistics, filename):
    with open(filename, 'w') as log_file:
        log_file.write(statistics)

def save_q_values(q, filename):
    np.save(filename, q)

def create_plots(run_number, iterations, 
                 average_num_containers, 
                 average_horizontal_scaling_final, 
                 average_vertical_scale_final, 
                 avarage_response_time_new,
                 average_cost_per_episode,
                 average_cpu_utilization):
    """Generates plots for the specified metrics."""

    # Number of containers per episode
    plot_metric(iterations, average_num_containers, 
                'Number of Containers', 'Containers per Episode', 
                f'/app/plots/num_containers_per_episode_{run_number}.png')

    # Number of vertical scaling actions per episode
    plot_metric(iterations, average_vertical_scale_final, 
                'Number of Vertical Scaling Actions', 'Vertical Scaling per Episode', 
                f'/app/plots/vertical_scaling_per_episode_{run_number}.png')

    # Number of horizontal scaling actions per episode
    plot_metric(iterations, average_horizontal_scaling_final, 
                'Number of Horizontal Scaling Actions', 'Horizontal Scaling per Episode', 
                f'/app/plots/horizontal_scaling_per_episode_{run_number}.png')

    # Average response time in milliseconds per episode
    plot_metric(iterations, np.array(avarage_response_time_new) * 1000, 
                'Response Time (ms)', 'Average Response Time per Episode', 
                f'/app/plots/avg_response_time_per_episode_{run_number}.png')

    # Average cost per episode
    plot_metric(iterations, average_cost_per_episode, 
                'Cost per Episode', 'Average Cost per Episode', 
                f'/app/plots/avg_cost_per_episode_{run_number}.png')

    # Average CPU utilization per episode
    plot_metric(iterations, average_cpu_utilization, 
                'CPU Utilization (%)', 'Average CPU Utilization per Episode', 
                f'/app/plots/avg_cpu_utilization_per_episode_{run_number}.png')

    print(f"Plots saved for run {run_number}")

def gather_learning_metrics_and_save(run_number, q, num_episodes, w_perf, w_res, w_adp, Rmax,
                                     avarage_response_time_new, average_horizontal_scaling_final,
                                     avarage_vertical_scale_final, average_cost_per_episode,
                                     average_cpu_utilization, horizontal_scaling_percentage,
                                     vertical_scaling_percentage):
    q_learning_log_path = f'/logs/q-learning-final-log_{run_number}.txt'
    q_learning_values_path = f'/logs/q-values_{run_number}.npy'
    q_learning_statistics = (
        f"Q-learning Final Episode Statistics:\n"
        f"Estimated Running Time: {(num_episodes * seconds_for_next_episode) / 60:.2f} minutes\n"
        f"Wperf = {w_perf}, Wres = {w_res}, Wadp = {w_adp}, Rmax = {Rmax}\n"
        f"Average Response Time: {avarage_response_time_new:.2f} s\n"
        f"Average Vertical Scale: {avarage_vertical_scale_final:.2f}%\n"
        f"Average Horizontal Scale: {average_horizontal_scaling_final:.2f}%\n"
        f"Total Horizontal Scaling Percentage: {horizontal_scaling_percentage:.2f}%\n"
        f"Total Vertical Scaling Percentage: {vertical_scaling_percentage:.2f}%\n"
        f"Average Cost per Episode: {average_cost_per_episode:.4f}\n"
        f"Average CPU Utilization per Episode: {average_cpu_utilization:.2f}%\n"
    )    
    save_final_statistics(q_learning_statistics, q_learning_log_path)
    save_q_values(q, q_learning_values_path)
    print(q_learning_statistics)
    
if __name__ == '__main__':
    
    file_path = 'Generated_Weights/q_learning_weights.txt'
    
    check_and_delete_file(file_path)
    
    w_perf_list, w_adp_list, w_res_list = create_file_with_random_weights(file_path, num_rows=20)    
    
    length = len(w_perf_list)
    
    print("Log: Generated weights:")
    
    for i in range(length):
        print(f"Log: (w_perf): {w_perf_list[i]:.2f}, (w_adp): {w_adp_list[i]:.2f}, (w_res): {w_res_list[i]:.2f}")
    
    for i in range(length):
        
        reset_environment_to_initial_state()
        
        # Initialize Q-table
        Q = np.zeros((len(state_space), len(action_space)))
        
        q_learning_metrics = run_q_learning(total_episodes, w_perf_list[i], w_adp_list[i], w_res_list[i])     
        
        (costs_per_episode,
        average_num_containers,
        avarage_response_time_new,
        average_horizontal_scaling_final,
        avarage_vertical_scale_final,
        average_cost_per_episode,
        average_cpu_utilization,
        w_adp, w_perf, w_res, q,
        horizontal_scaling_percentage,
        vertical_scaling_percentage) = q_learning_metrics
        num_iterations = len(costs_per_episode)
        
        iterations = range(1, num_iterations + 1)
        
        running_time = total_episodes * seconds_for_next_episode / 60
        
        create_plots(run_number=i, 
                    iterations=iterations, 
                    average_num_containers=average_num_containers,
                    average_horizontal_scaling_final=average_horizontal_scaling_final, 
                    average_vertical_scale_final=avarage_vertical_scale_final, 
                    avarage_response_time_new=avarage_response_time_new,
                    average_cost_per_episode=average_cost_per_episode,
                    average_cpu_utilization=average_cpu_utilization)
        
        gather_learning_metrics_and_save(i, q, total_episodes, w_perf, w_res, w_adp, Rmax,
                                 avarage_response_time_new, average_horizontal_scaling_final,
                                 avarage_vertical_scale_final, average_cost_per_episode,
                                 average_cpu_utilization, horizontal_scaling_percentage,
                                 vertical_scaling_percentage)

    reset_environment_to_initial_state()
    
    if baseline:
        # Run baseline scenario (without Q-learning) and gather metrics
        baseline_metrics = run_baseline(total_episodes)
        
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
            f"Rmax Violations: {Rmax_violations[-1] * 100 / total_episodes:.2f}%\n"
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
