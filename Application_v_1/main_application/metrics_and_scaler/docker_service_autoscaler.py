import numpy as np
import random, time, docker, logging
import prometheus_metrics

url = 'http://prometheus:9090/api/v1/query'
cpu_threshold = 20
ram_threshold = 15
service_name = 'mystack_mnist'
max_replicas = 7
min_replicas = 1
num_states = 10
Q = np.zeros((num_states, num_states, 2))

def discretize_state(cpu_value, ram_value):
    cpu_state = int(cpu_value / 10)
    ram_state = int(ram_value / 10)
    return cpu_state, ram_state

def select_action(state):
    epsilon = 0.1
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1])
    else:
        return np.argmax(Q[state[0], state[1], :])

def update_q_value(state, action, reward, next_state):
    alpha = 0.1
    gamma = 0.9
    Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action])

def get_reward(cpu_value, ram_value):
    if cpu_value <= cpu_threshold and ram_value <= ram_threshold:
        return 1
    else:
        return 0

def scale_out(service_name, desired_replicas):
    client = docker.from_env()
    service = client.services.get(service_name)
    service.scale(desired_replicas)
    print(f"Service '{service_name}' scaled to {desired_replicas} replicas.")

def scale_in(service_name, scale_out_factor):
    client = docker.from_env()
    service = client.services.get(service_name)
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is not None:
        print(f"Service '{service_name}' not found.")
        return
    desired_replicas = min(current_replicas + scale_out_factor, max_replicas)
    service.scale(desired_replicas)

def get_current_replica_count(service_name):
    client = docker.from_env()
    try:
        service = client.services.get(service_name)
        return service.attrs['Spec']['Mode']['Replicated']['Replicas']
    except docker.errors.NotFound:
        return None

def fetch_data():
    try:
        metrics = prometheus_metrics.start_metrics_service(True, url)
        if metrics is not None:
            time_up = metrics[2]
            if time_up != '0':
                cpu_percent = int(float(metrics[0]))
                ram_percent = int(float(metrics[1]))
                time_up = int(float(time_up))
                return cpu_percent, ram_percent, time_up
        print("No metrics available, wait...")
    except Exception as e:
        print("An error occurred during service CPU retrieval:", e)
    return None, None, None

def main():
    logger = logging.getLogger(__name__) 
    # while True:
    cpu_value, ram_value, _ = fetch_data()
    if cpu_value is not None and ram_value is not None:
        cpu_state, ram_state = discretize_state(cpu_value, ram_value)
        state = (cpu_state, ram_state)
        action = select_action(state)
    else:
        action = -1
    if action == 0:
        current_replicas = get_current_replica_count(service_name)
        if current_replicas is not None and current_replicas < max_replicas:
            scale_out(service_name, current_replicas + 1)
            reward = get_reward(cpu_value, ram_value)
            next_cpu_value, next_ram_value, _ = fetch_data()
            next_cpu_state, next_ram_state = discretize_state(next_cpu_value, next_ram_value)
            next_state = (next_cpu_state, next_ram_state)
            update_q_value(state, action, reward, next_state)
            logger.info(f"Horizontal Scale Out: Replicas increased to {current_replicas + 1}")
            print("REWARD ---->",reward)
        else:
            logger.info(f"Already at maximum replicas: {max_replicas}")
    elif action == 1:
        current_replicas = get_current_replica_count(service_name)
        if current_replicas is not None and current_replicas > min_replicas:
            scale_in(service_name, 1)
            reward = get_reward(cpu_value, ram_value)
            next_cpu_value, next_ram_value, _ = fetch_data()
            next_cpu_state, next_ram_state = discretize_state(next_cpu_value, next_ram_value)
            next_state = (next_cpu_state, next_ram_state)
            update_q_value(state, action, reward, next_state)
            logger.info(f"Horizontal Scale In: Replicas decreased to {current_replicas - 1}")
        else:
            logger.info(f"Already at minimum replicas: {min_replicas}")
    else:
            logger.info("No action taken")

if __name__ == "__main__":
    while True:
        main()