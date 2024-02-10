import numpy as np
import random, logging, docker
import prometheus_metrics
import matplotlib.pyplot as plt
import time
from datetime import datetime
from autoscaler_env import AutoscaleEnv
import skfuzzy as fuzz

url = 'http://prometheus:9090/api/v1/query'
service_name = 'mystack_application'
application_url = 'http://application:8501/train'
cpu_threshold = 8
ram_threshold = 10
max_replicas = 10
min_replicas = 1
num_states = 2
# Q_file = "q_values.npy"
# Q = np.load(Q_file) if Q_file else np.zeros((num_states, num_states, 2))
Q = np.zeros((num_states, num_states, 2))
iteration = 1

# Define fuzzy input variables (Universe and membership functions)
cpu = np.arange(0, 101)
ram = np.arange(0, 101)

# Define fuzzy output variable (Universe and membership functions)
# action = np.arange(0, 2)

cpu_high = fuzz.trimf(cpu, [50, 100, 100])
cpu_low = fuzz.trimf(cpu, [0, 0, 50])
ram_high = fuzz.trimf(ram, [50, 100, 100])
ram_low = fuzz.trimf(ram, [0, 0, 50])

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

def select_action(Q, cpu_state, ram_state):
    """
    Selects an action based on the given state.

    Args:
        Q (numpy.ndarray): The Q-values.
        cpu_state (int): The CPU state.
        ram_state (int): The RAM state.

    Returns:
        int: The selected action.
    """
    epsilon = 0.1
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1])
    else:
        cpu_state = min(max(cpu_state, 0), num_states - 1)
        ram_state = min(max(ram_state, 0), num_states - 1)
        return np.argmax(Q[cpu_state, ram_state, :])

def update_q_value(Q, state, action, reward, next_state):
    """
    Updates the Q-value based on the given state, action, reward, and next state.

    Args:
        Q (numpy.ndarray): The Q-values.
        state (tuple): A tuple representing the current state.
        action (int): The selected action.
        reward (int): The reward received for the action.
        next_state (tuple): A tuple representing the next state.
    """
    alpha = 0.6
    gamma = 0.5
        
    Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (
        reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action]
    )

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
    
def get_docker_services():
    try:
        # Connect to the Docker daemon
        client = docker.from_env()

        # Get a list of all services
        services = client.services.list()

        # Extract relevant information from each service
        service_info = []
        for service in services:
            service_info.append({
                'ID': service.id,
                'Name': service.name,
                'Replicas': service.attrs['Spec']['Mode']['Replicated']['Replicas'],
                'Image': service.attrs['Spec']['TaskTemplate']['ContainerSpec']['Image'],
                'Ports': service.attrs.get('Endpoint', {}).get('Ports', [])
            })
        return service_info

    except docker.errors.APIError as e:
        print(f"Error connecting to Docker: {e}")
        return None
# This code implementing a control loop that monitors CPU and RAM metrics, takes certain actions based on thresholds and conditions, and updates a Q-value. 
# It also handles scaling operations based on the current state and actions.
if __name__ == "__main__":
    """
    The main method for this project
    """
    env = AutoscaleEnv(service_name, min_replicas, max_replicas, cpu_threshold, ram_threshold, num_states)
    # while True:
    #     docker_services = get_docker_services()
    #     if docker_services is not None:
    #         for service in docker_services:
    #             print(service)
    #     time.sleep(10)
    Run_using_fuzzy = True
    train_steps = 200
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"Application Information:\nStart date & time:{dt_string}\nObservable service name:{service_name}\nContext urls:{url}, {application_url},\nTrain Steps = {train_steps}\n")
    mse_values = [] # Initialize empty list to store MSE values
    rewards = []
    replicas_count = []
    save_path = '/plots' # Set the save path inside the container
    validation_interval = train_steps - 1 # Perform validation every x iterations
    obs = env.reset()
    for iteration in range(1, train_steps):  # Run iterations
        logger = logging.getLogger(__name__)  # Initialize a logger
        print(f"\n--------Iteration No:{iteration}")  # Print the current iteration number
        logger.setLevel(logging.DEBUG)  # Set the logger level to debug
        handler = logging.StreamHandler()  # Create a stream handler for the logger
        cpu_value, ram_value, time_running = fetch_data()  # Get CPU, RAM values and a placeholder value
        
        # Validation and Evaluation
        if iteration % validation_interval == 0:
            print("\n------Validation phase starts. Please wait until Validation Phase stop")
            validation_rewards = []
            for _ in range(50):  # Run 50 validation episodes
                # Use the validation environment (not the training environment)
                validation_obs = env.reset()
                validation_reward = 0
                while True:
                    validation_action = select_action(Q, *discretize_state(*validation_obs))
                    validation_next_obs, validation_reward, validation_done, _ = env.step(validation_action)
                    
                    validation_obs = validation_next_obs
                    if validation_done:
                        break

                validation_rewards.append(validation_reward)

            avg_validation_reward = np.mean(validation_rewards)
            print(f"\nValidation Iteration: {iteration}, Avg. Validation Reward: {avg_validation_reward}\n Validation phase complete")
        
        # Check if CPU and RAM values are not None
        if cpu_value is not None and ram_value is not None:
            # cpu_value
            print(f"Metrics: |CPU:{str(cpu_value)}% |RAM:{str(ram_value)} % |Time running:{str(time_running)}s")  # Print metrics
            
            # Check if CPU or RAM exceed thresholds
            if cpu_threshold < cpu_value or ram_threshold < ram_value:
                # Check if CPU and RAM values are not None
                if cpu_value is not None and ram_value is not None:
                    # Discretize states
                    cpu_state, ram_state = discretize_state(float(cpu_value), float(ram_value))
                    if Run_using_fuzzy is not True:
                        # Select an action based on your Q-learning logic
                        action = select_action(Q, cpu_state, ram_state)
                    if Run_using_fuzzy:
                        # Fuzzy Inference for scaling out
                        rule1 = fuzz.relation_min(fuzz.trimf(cpu, [cpu_value - 10, cpu_value, cpu_value + 10])[:, np.newaxis],
                                                fuzz.trimf(ram, [ram_value - 10, ram_value, ram_value + 10])[:, np.newaxis])
                        action_scale_out = np.array([1, 0])
                        action_activation_scale_out = np.fmin(rule1[:, :, np.newaxis], action_scale_out)

                        # Fuzzy Inference for scaling in (corrected)
                        rule2 = fuzz.relation_min(fuzz.trimf(cpu, [cpu_value - 10, cpu_value, cpu_value + 10])[:, np.newaxis],
                                                fuzz.trimf(ram, [ram_value - 10, ram_value, ram_value + 10])[:, np.newaxis])
                        action_scale_in = np.array([0, 1])
                        action_activation_scale_in = np.fmin(rule2[:, :, np.newaxis], action_scale_in)

                        # Combine the fuzzy inferences
                        aggregated = np.fmax(action_activation_scale_out, action_activation_scale_in)

                        # Defuzzification
                        universe_of_discourse = np.arange(0, 3)
                        max_index = np.unravel_index(np.argmax(aggregated), aggregated.shape)
                        print(f"Max-Index = {max_index}")
                        action_result = universe_of_discourse[max_index[2]]
                        print(f"Action_result={action_result}")
                        selected_action = int(round(action_result))
                        print("Selected Action:", selected_action)
                        if action_result >= 0.5:
                            # Scale out action
                            action = 0
                        else:
                            # Scale in action
                            action = 1

                    # Take a step in the environment
                    next_obs, reward, done, _ = env.step(action)
                    next_cpu_state, next_ram_state = discretize_state(float(next_obs[0]), float(next_obs[1]))
                    next_state = (next_cpu_state, next_ram_state)
                    update_q_value(Q, (cpu_state, ram_state), action, reward, (next_cpu_state, next_ram_state))
                    iteration += 1  # Increment iteration count
                    # Log Q-values
                    print(f"Iteration: {iteration}, Q-values: \n{Q}")
                    # Log rewards
                    print(f"Iteration: {iteration}, Reward: {reward}")
                    # Calculate MSE and store in list
                    target_values = np.array([[cpu_state, ram_state]])  # Use the current state as target values
                    mse = calculate_mse(Q, target_values) # Calculate MSE
                    mse_values.append(mse)

                    if iteration % 100 == 0:  # Save plot every 100 iterations
                        print("Plotting...")
                        plot_values(range(1, iteration+1, 10), mse_values[::10], save_path)
                
            else:  # If CPU and RAM do not exceed thresholds
                # Check if CPU and RAM values are not None
                if cpu_value is not None and ram_value is not None:
                    print(f"Metrics: |CPU:{str(cpu_value)}% |RAM:{str(ram_value)} % |Time running:{str(time_running)}s")  # Print metrics
                    # Discretize states
                    cpu_state, ram_state = discretize_state(float(cpu_value), float(ram_value))
                    if Run_using_fuzzy is not True:
                        # Select an action based on your Q-learning logic
                        action = select_action(Q, cpu_state, ram_state)
                    if Run_using_fuzzy:
                        # Fuzzy Inference
                        rule1 = fuzz.relation_min(cpu_high[:, np.newaxis], ram_high[:, np.newaxis])
                        action_scale_out = np.array([1, 0])
                        action_activation_scale_out = np.fmin(rule1[:, :, np.newaxis], action_scale_out)
                        rule2 = np.fmax(cpu_low[:, np.newaxis], ram_low[:, np.newaxis])
                        action_scale_in = np.array([0, 1])
                        action_activation_scale_in = np.fmin(rule2[:, :, np.newaxis], action_scale_in)
                        aggregated = np.fmax(action_activation_scale_out, action_activation_scale_in)
                        # Defuzzification
                        universe_of_discourse = np.array([0, 1])
                        max_index = np.unravel_index(np.argmax(aggregated), aggregated.shape)
                        print(f"Max-Index = {max_index}")
                        action_result = universe_of_discourse[max_index[1]]
                        print(f"Action_result={action_result}")
                        selected_action = int(round(action_result))
                        print("Selected Action:", selected_action)
                        if action_result >= 0.5:
                            # Scale out action
                            action = 0
                        else:
                            # Scale in action
                            action = 1
                    # Take a step in the environment
                    next_obs, reward, done, _ = env.step(action)
                    next_cpu_state, next_ram_state = discretize_state(float(next_obs[0]), float(next_obs[1]))
                    next_state = (next_cpu_state, next_ram_state)
                    update_q_value(Q, (cpu_state, ram_state), action, reward, (next_cpu_state, next_ram_state))
                    iteration += 1  # Increment iteration count
                    # Log Q-values
                    print(f"Iteration: {iteration}, Q-values: \n{Q}")
                    # Log rewards
                    print(f"Iteration: {iteration}, Reward: {reward}")
                    # Calculate MSE and store in list
                    target_values = np.array([[cpu_state, ram_state]])  # Use the current state as target values
                    mse = calculate_mse(Q, target_values) # Calculate MSE
                    mse_values.append(mse)
                    if iteration % 10 == 0:  # Save plot every 50 iterations
                        print("Plotting...")
                        plot_values(range(1, iteration+1, 10), mse_values[::10], save_path)
            np.save('/QSavedWeights/q_values.npy', Q)
        else:  # If CPU or RAM values are None
            print("No metrics available, wait...")  # Indicate no metrics available