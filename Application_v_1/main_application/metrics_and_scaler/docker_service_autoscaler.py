import numpy as np
import random, time, docker, logging
import prometheus_metrics

url = 'http://prometheus:9090/api/v1/query'
cpu_threshold = 8
ram_threshold = 20
service_name = 'mystack_application'
max_replicas = 7
min_replicas = 1
num_states = 10
Q = np.zeros((num_states, num_states, 2))
iteration = 1

def discretize_state(cpu_value, ram_value): #TO-DO Needs check.
    cpu_state = int(cpu_value / 10)
    ram_state = int(ram_value / 10)
    
    cpu_state = max(0, min(cpu_state, num_states - 1))
    ram_state = max(0, min(ram_state, num_states - 1))

    return cpu_state, ram_state

def select_action(state):
    epsilon = 0.8
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1])
    else:
        cpu_state = min(max(state[0], 0), num_states - 1)
        ram_state = min(max(state[1], 0), num_states - 1)
        return np.argmax(Q[cpu_state, ram_state, :])

def update_q_value(state, action, reward, next_state):
    alpha = 0.1
    gamma = 0.9
    Q[state[0], state[1], action] = Q[state[0], 
                                      state[1], 
                                      action] + alpha * (reward + gamma * np.max(Q[next_state[0], 
                                                                                   next_state[1], 
                                                                                   :]) - Q[state[0], 
                                                                                           state[1], 
                                                                                           action])

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

def get_current_replica_count(service_prefix):
    client = docker.from_env()
    try:
        for service in client.services.list():
            if service_prefix in service.name:
                return service.attrs['Spec']['Mode']['Replicated']['Replicas']
        return None
    except docker.errors.NotFound:
        return None

def fetch_data():
    try:
        metrics = prometheus_metrics.start_metrics_service(url = url)
        if metrics is not None:
            time_up = metrics[2]
            if time_up != '0':
                cpu_percent = int(float(metrics[0]))
                ram_percent = int(float(metrics[1]))
                time_up = int(float(time_up))
                return cpu_percent, ram_percent, time_up
        elif metrics is None:
            return None, None, None
    except Exception as e:
        print("An error occurred during service metrics retrieval:", e)
    

if __name__ == "__main__":
    print("Script is running...")
    while True:
        time.sleep(5)
        logger = logging.getLogger(__name__)
        print("--------Iteration No:", str(iteration))
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        cpu_value, ram_value, _ = fetch_data()
        if cpu_value is not None and ram_value is not None:
            print(f"Metrics: |CPU:{str(cpu_value)}% |RAM:{str(ram_value)} % |Time running:{str(_)}s")
            if  cpu_threshold < cpu_value or ram_threshold < ram_value:
                if cpu_value is not None and ram_value is not None:
                    cpu_state, ram_state = discretize_state(float(cpu_value), float(ram_value))
                    print(f"Discretized values: cpu={cpu_state} ram={ram_state}")
                    state = (cpu_state, ram_state)
                    action = select_action(state)
                    logger.info(f"State: {action}")
                    print(f"Action: {action}")
                else:
                    logger.info(f"CPU & RAM values did not return any data: {cpu_value, ram_value}")
                    print("CPU & RAM values did not return any data...")
                if action == 0:
                    current_replicas = get_current_replica_count(service_name)
                    if current_replicas is not None and current_replicas < max_replicas:
                        #scale_out(service_name, current_replicas + 1)
                        reward = get_reward(cpu_value, ram_value)
                        next_cpu_value, next_ram_value, _ = fetch_data()
                        next_cpu_state, next_ram_state = discretize_state(next_cpu_value, next_ram_value)
                        next_state = (next_cpu_state, next_ram_state)
                        update_q_value(state, action, reward, next_state)
                        logger.info(f"Horizontal Scale Out: Replicas increased to {current_replicas + 1}")
                        print(f"Horizontal Scale Out: Replicas increased to: {current_replicas}")
                        print("REWARD ---->",reward)
                    else:
                        logger.info(f"Already at maximum replicas: {max_replicas}")
                        print("Already at maximum replicas")
                elif action == 1:
                    current_replicas = get_current_replica_count(service_name)
                    if current_replicas is not None and current_replicas > min_replicas:
                        #scale_in(service_name, 1)
                        reward = get_reward(cpu_value, ram_value)
                        next_cpu_value, next_ram_value, _ = fetch_data()
                        next_cpu_state, next_ram_state = discretize_state(next_cpu_value, next_ram_value)
                        next_state = (next_cpu_state, next_ram_state)
                        update_q_value(state, action, reward, next_state)
                        logger.info(f"Horizontal Scale In: Replicas decreased to {current_replicas - 1}")
                        print(f"Horizontal Scale In: Replicas decreased to:{current_replicas}")
                    else:
                        logger.info(f"Already at minimum replicas: {min_replicas}")
                        print("Already at minimum replicas")
                else:
                    logger.info("No action taken")
                    print("No action taken")
                iteration += 1
            else:
                logger.info("Logger Level(info): No action taken, because the cpu value is not greater than cpu or ram threshold.")
                print("No action taken, because the cpu value or ram value is not greater than cpu or ram threshold.")
                iteration += 1
        else:
            print("No metrics available, wait...")  
