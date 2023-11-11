import numpy as np
import random, time, docker, logging, itertools, requests
import prometheus_metrics
import matplotlib.pyplot as plt

url = 'http://prometheus:9090/api/v1/query'
cpu_threshold = 8
ram_threshold = 20
service_name = 'mystack_application'
max_replicas = 7
min_replicas = 1
num_states = 2
Q = np.zeros((num_states, num_states, 2))
iteration = 1
reward = 0
cpu_values = []
ram_values = []

def discretize_state(cpu_value, ram_value):
    """
    Discretizes CPU and RAM values.

    Args:
        cpu_value (float): The CPU value.
        ram_value (float): The RAM value.

    Returns:
        tuple: A tuple containing the discretized CPU and RAM states.
    """
    cpu_state = int(cpu_value)
    ram_state = int(ram_value)
    
    cpu_state = max(0, min(cpu_state, num_states - 1))
    ram_state = max(0, min(ram_state, num_states - 1))

    return cpu_state, ram_state

def select_action(state):
    """
    Selects an action based on the given state.

    Args:
        state (tuple): A tuple representing the current state.

    Returns:
        int: The selected action.
    """
    epsilon = 0.1
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1])
    else:
        cpu_state = min(max(state[0], 0), num_states - 1)
        ram_state = min(max(state[1], 0), num_states - 1)
        return np.argmax(Q[cpu_state, ram_state, :])

def update_q_value(state, action, reward, next_state):
    """
    Updates the Q-value based on the given state, action, reward, and next state.

    Args:
        state (tuple): A tuple representing the current state.
        action (int): The selected action.
        reward (int): The reward received for the action.
        next_state (tuple): A tuple representing the next state.
    """
    alpha = 0.7
    gamma = 0.5
    Q[state[0], state[1], action] = Q[state[0],
                                      state[1],
                                      action] + alpha * (reward + gamma * np.max(Q[next_state[0],
                                                                                   next_state[1],
                                                                                   :]) - Q[state[0],
                                                                                           state[1],
                                                                                           action])

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
        if cpu_value <= cpu_threshold or ram_value <= ram_threshold:
            return 20
        else:
            return -10
    else:
        return 0

def scale_out(service_name, desired_replicas):
    """
    Scales out a service to the specified number of replicas.
    """
    client = docker.from_env()
    service = client.services.get(service_name)
    service.scale(desired_replicas)
    print(f"Service '{service_name}' scaled to {desired_replicas} replicas.")

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
    time.sleep(60)

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

def Calculate_Thresholds():
    """
    Calculates the CPU and RAM thresholds.

    Returns:
        tuple: A tuple containing the CPU and RAM thresholds.
    """
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is not None:
        cpu_threshold = 20 + (current_replicas - 1) * 10 if current_replicas <= 6 else 90
        ram_threshold = 20 + (current_replicas - 1) * 10 if current_replicas <= 6 else 90
    else:
        cpu_threshold = 10  # Default value if replicas count is not available
        ram_threshold = 20  # Default value if replicas count is not available

    print(f"Thresholds calculated as CPU:{cpu_threshold}, RAM: {ram_threshold}")
    return cpu_threshold, ram_threshold

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

def plot_values(iterations, mse_values, save_path):
    """
    Plots Mean Squared Error (MSE) over iterations.
    
    Args:
        iterations (list): List of iteration numbers.
        mse_values (list): List of MSE values.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, mse_values, label='MSE')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(f'{save_path}/mse_plot_iteration_{iteration}.png')
    
# This code implementing a control loop that monitors CPU and RAM metrics, takes certain actions based on thresholds and conditions, and updates a Q-value. 
# It also handles scaling operations based on the current state and actions.
if __name__ == "__main__":
    """
    The main method for this project
    """
    print("Script is running...")  # Indicate that the script is running
    mse_values = [] # Initialize empty list to store MSE values
    rewards = []
    replicas_count = []
    save_path = '/plots' # Set the save path inside the container
    while True:  # Run an infinite loop
        time.sleep(5)  # Pause for 5 seconds
        logger = logging.getLogger(__name__)  # Initialize a logger
        print(f"\n--------Iteration No:{iteration}")  # Print the current iteration number
        logger.setLevel(logging.DEBUG)  # Set the logger level to debug
        handler = logging.StreamHandler()  # Create a stream handler for the logger
        cpu_value, ram_value, time_running = fetch_data()  # Get CPU, RAM values and a placeholder value
        
        # Check if CPU and RAM values are not None
        if cpu_value is not None and ram_value is not None:
            # cpu_value
            print(f"Metrics: |CPU:{str(cpu_value)}% |RAM:{str(ram_value)} % |Time running:{str(time_running)}s")  # Print metrics
            
            # Check if CPU or RAM exceed thresholds
            if cpu_threshold < cpu_value or ram_threshold < ram_value:
                # Check if CPU and RAM values are not None
                if cpu_value is not None and ram_value is not None:
                    #distribute_request(service_name)  # Distribute request
                    cpu_state, ram_state = discretize_state(float(cpu_value), float(ram_value))  # Discretize states
                    print(f"Discretized values: cpu={cpu_state} ram={ram_state}")  # Print discretized values
                    state = (cpu_state, ram_state)  # Define state tuple
                    action = select_action(state)  # Select an action
                    logger.info(f"State: {action}")  # Log state
                    print(f"Action: {action}")  # Print action
                else:
                    logger.info(f"CPU & RAM values did not return any data: {cpu_value, ram_value}")  # Log absence of data
                    print("CPU & RAM values did not return any data...")  # Print absence of data
                
                if action == 0:  # If action is 0 (scale out)
                    # Get current replica count
                    current_replicas = get_current_replica_count(service_name)
                    # If current replicas is not None and less than max replicas
                    if current_replicas is not None and current_replicas < max_replicas:
                        replicas_count.append(int(current_replicas))
                        scale_out(service_name, current_replicas + 1)  # Increase replicas
                        print(f"Horizontal Scale Out: Replicas increased to: {current_replicas}, system waits 5 seconds...")  # Print scale out message
                        time.sleep(5)  # Wait for 5 seconds
                        tuple_data = fetch_data()  # Fetch new data
                        has_data = all(ele is None for ele in tuple_data)  # Check if data is available
                        if not has_data:
                            cpu_value, ram_value, time_running = tuple_data  # Get new CPU and RAM values
                            print("Calculating reward...")  # Indicate reward calculation
                            time.sleep(1)  # Wait for 1 second
                            cpu_threshold, ram_threshold = Calculate_Thresholds()  # Get dynamic thresholds
                            print(f"Thresholds:{cpu_threshold, ram_threshold} and Cpu & Ram Values:{cpu_value, ram_value}")  # Print thresholds and values
                            reward = get_reward(cpu_value, ram_value, cpu_threshold, ram_threshold)  # Calculate reward using dynamic thresholds
                            rewards.append(reward)
                            print(f"Reward was: {reward} with cpu val: {cpu_value} and ram val: {ram_value}")  # Print reward
                            next_cpu_value, next_ram_value, time_running = fetch_data()  # Get next CPU and RAM values
                            if next_cpu_value is not None and next_ram_value is not None and time_running is not None:
                                next_cpu_state, next_ram_state = discretize_state(next_cpu_value, next_ram_value)  # Discretize next state
                                next_state = (next_cpu_state, next_ram_state)  # Define next state
                                update_q_value(state, action, reward, next_state)  # Update Q-value
                                logger.info(f"Horizontal Scale Out: Replicas increased to {current_replicas + 1}")  # Log scale out
                            else:
                                print("CPU or RAM value is None. Skipping this iteration.")
                        else:
                            print("Something went wrong when trying to fetch the data for calculation the reward. System will retry in 30 seconds...")  # Indicate data retrieval issue
                            time.sleep(30)  # Wait for 30 seconds
                    else:
                        logger.info(f"Already at maximum replicas: {max_replicas}")  # Log already at maximum replicas
                        print("Already at maximum replicas. Resetting replicas to 1")  # Indicate already at maximum replicas
                        reset_replicas(service_name=service_name)
                elif action == 1:  # If action is 1 (scale in)
                    current_replicas = get_current_replica_count(service_name)  # Get current replica count
                    if current_replicas is not None and current_replicas > min_replicas:  # If current replicas is not None and greater than min replicas
                        replicas_count.append(int(current_replicas))
                        scale_in(service_name, 1)  # Decrease replicas
                        print(f"Horizontal Scale In: Replicas decreased to: {current_replicas -1}, system waits 5 seconds")  # Print scale in message
                        time.sleep(5)  # Wait for 5 seconds
                        tuple_data = fetch_data()  # Fetch new data
                        has_data = all(ele is None for ele in tuple_data)  # Check if data is available
                        if not has_data:
                            cpu_value, ram_value, time_running = tuple_data
                            print("Calculating reward...")  # Indicate reward calculation
                            time.sleep(1)  # Wait for 1 second
                            cpu_threshold, ram_threshold = Calculate_Thresholds()  # Get dynamic thresholds
                            print(f"Thresholds:{cpu_threshold, ram_threshold} and Cpu & Ram Values:{cpu_value, ram_value}")  # Print thresholds and values
                            reward = get_reward(cpu_value, ram_value, cpu_threshold, ram_threshold)  # Calculate reward using dynamic thresholds
                            rewards.append(reward)
                            print(f"Reward was: {reward} with cpu val: {cpu_value} and ram val: {ram_value}")  # Print reward
                            next_cpu_value, next_ram_value, time_running = fetch_data()  # Get next CPU and RAM values
                            if next_cpu_value is not None and next_ram_value is not None and time_running is not None:
                                next_cpu_state, next_ram_state = discretize_state(next_cpu_value, next_ram_value)  # Discretize next state
                                next_state = (next_cpu_state, next_ram_state)  # Define next state
                                update_q_value(state, action, reward, next_state)  # Update Q-value
                                print(f"Horizontal Scale In: Replicas decreased to:{current_replicas}")  # Print new replica count
                            else:
                                print("CPU or RAM value is None. Skipping this iteration.")
                        else:
                            print("Something went wrong when trying to fetch the data for calculation the reward. System will retry in 30 seconds...")  # Indicate data retrieval issue
                            time.sleep(30)  # Wait for 30 seconds
                    else:
                        logger.info(f"Already at minimum replicas: {min_replicas}")  # Log already at minimum replicas
                        print("Already at minimum replicas")  # Indicate already at minimum replicas
                else:
                    logger.info("No action taken")  # Log no action taken
                    print("No action taken")  # Indicate no action taken
                iteration += 1  # Increment iteration count
                
                # Calculate MSE and store in list
                target_values = np.array([[cpu_state, ram_state]])  # Use the current state as target values
                mse = calculate_mse(Q, target_values) #Calculate MSE
                mse_values.append(mse)

                if iteration % 10 == 0:  # Save plot every 10 iterations
                    print("Plotting...")
                    plot_values(range(1, iteration+1, 10), mse_values[::10], save_path)
                
            else:  # If CPU and RAM do not exceed thresholds
                current_replicas = get_current_replica_count(service_name)  # Get current replica count
                
                # If current replicas is greater than min replicas
                if current_replicas > min_replicas:
                    scale_in(service_name, 1)  # Decrease replicas
                    print(f"Removing replicas because there is no demand, current replicas: {current_replicas}")  # Print removal of replicas
                    iteration += 1  # Increment iteration count
                    cpu_threshold = 8  # Set CPU threshold to 8
                    ram_threshold = 20  # Set RAM threshold to 20
                    time.sleep(5)  # Wait for 5 seconds
                else:
                    logger.info("Logger Level(info): No action taken, because the cpu value is not greater than cpu or ram threshold.")  # Log no action taken due to thresholds
                    print("No action taken, because the cpu value or ram value is not greater than cpu or ram threshold.")  # Indicate no action taken due to thresholds
                    iteration += 1  # Increment iteration count
        else:  # If CPU or RAM values are None
            print("No metrics available, wait...")  # Indicate no metrics available