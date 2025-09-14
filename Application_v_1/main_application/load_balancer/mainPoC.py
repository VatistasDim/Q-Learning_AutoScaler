import numpy as np
import random
import time
from scalingOperations import (
    scale_out,
    scale_in,
    set_cpu_shares,
    calculate_cpu_shares,
    get_current_replica_count,
    get_current_cpu_shares
)
from costs import Costs
from docker_api import DockerAPI
from prometheus_metrics import start_metrics_service

# ----------------------------
# Parameters
# ----------------------------
Kmax = 10        # maximum containers
u_max = 100      # CPU utilization (%)
c_max = 100      # CPU shares
u_quantum = 10
c_quantum = 10

docker_api = DockerAPI(stack_name="mystack_application")

# Q-learning parameters
episodes = 10
steps_per_episode = 30
alpha = 0.1
gamma = 0.95
epsilon = 0.1

# Scaling & cost parameters
Rmax = 200  # max acceptable response time
w_adp = 0.2
w_perf = 0.5
w_res = 0.3

# Prometheus URL
PROM_URL = "http://your-prometheus-server:9090/api/v1/query"

# ----------------------------
# State space
# ----------------------------
def get_state_space():
    states = []
    for k in range(1, Kmax + 1):
        for u in range(0, u_max + 1, u_quantum):
            for c in range(c_quantum, c_max + 1, c_quantum):
                states.append((k, u, c))
    return states

states = get_state_space()
state_to_idx = {s: i for i, s in enumerate(states)}

# ----------------------------
# Actions
# ----------------------------
actions = [("vscale", -10), ("hscale", -1), ("noop", 0), ("hscale", +1), ("vscale", +10)]
n_actions = len(actions)

# ----------------------------
# Helper functions
# ----------------------------
def discretize(value, quantum, v_min, v_max):
    value = max(v_min, min(value, v_max))
    return int(round(value / quantum) * quantum)

def fetch_data(service_name, url, max_attempts=10):
    for attempt in range(max_attempts):
        try:
            cpu_percent, ram_percent, time_up, response_time, cpu_shares = start_metrics_service(url)
            if time_up != '0' and None not in (cpu_percent, ram_percent, time_up, response_time, cpu_shares):
                cpu_percent = int(float(cpu_percent))
                ram_percent = int(float(ram_percent))
                time_up = int(float(time_up))
                response_time = float(response_time)
                cpu_shares = calculate_cpu_shares(get_current_cpu_shares(service_name))
                return cpu_percent, ram_percent, time_up, response_time, cpu_shares
        except Exception:
            time.sleep(2)
    return None, None, None, None, None

def apply_action(service_name, state, action, prometheus_url=None):
    k, u, c = state

    # Execute scaling
    if action[0] == "hscale":
        if action[1] > 0:
            scale_out(service_name, action[1])
        elif action[1] < 0:
            scale_in(service_name, abs(action[1]))
    elif action[0] == "vscale":
        new_cpu = c + action[1]
        new_cpu = max(c_quantum, min(new_cpu, c_max))
        set_cpu_shares(service_name, calculate_cpu_shares(new_cpu / 10))
    # noop does nothing

    # Update state with live metrics if available
    if prometheus_url:
        cpu_percent, _, _, response_time, cpu_shares = fetch_data(service_name, prometheus_url)
        if cpu_percent is not None:
            u = cpu_percent
        if cpu_shares is not None:
            c = cpu_shares
    else:
        u = random.randint(20, 80)
        response_time = 100 + (u * 2) - (k * 5) - (c * 0.2)

    k = get_current_replica_count(service_name)
    k = min(max(1, k), Kmax)
    u = min(max(0, u), u_max)
    c = min(max(c_quantum, c), c_max)
    time.sleep(2)
    # Estimate response time if Prometheus not available
    if prometheus_url is None or response_time is None:
        response_time = 100 + (u * 2) - (k * 5) - (c * 0.2)

    return (k, u, c), response_time

# ----------------------------
# Q-learning loop
# ----------------------------
Q = np.zeros((len(states), n_actions))

for ep in range(episodes):
    # Initial state
    state = (
        min(max(1, get_current_replica_count("mystack_application")), Kmax),
        random.randint(20, 80),
        min(max(c_quantum, get_current_cpu_shares("mystack_application")), c_max)
    )

    total_cost = 0

    for _ in range(steps_per_episode):
        s_idx = state_to_idx[state]

        # Epsilon-greedy
        a_idx = np.random.randint(n_actions) if random.random() < epsilon else np.argmin(Q[s_idx, :])
        action = actions[a_idx]

        # Apply action with live metrics
        next_state, R_current = apply_action("mystack_application", state, action, prometheus_url=PROM_URL)
        if R_current is None:
            R_current = 100 + (state[1]*2) - (state[0]*5) - (state[2]*0.2)

        # Extract action effect for cost function
        a1 = action[1] if action[0] == "hscale" else 0
        a2 = action[1] if action[0] == "vscale" else 0

        cost = Costs.overall_cost_function(
            wadp=w_adp, wperf=w_perf, wres=w_res,
            k_next_state=next_state[0],
            u_next_state=next_state[1],
            c_next_state=next_state[2],
            action=action,  # pass the tuple directly
            a1=a1, a2=a2,
            Rmax=Rmax,
            Kmax=Kmax,
            R=R_current
        )

        reward = -cost
        total_cost += cost

        # Q-update
        s_next_idx = state_to_idx[next_state]
        Q[s_idx, a_idx] = (1 - alpha) * Q[s_idx, a_idx] + alpha * (reward + gamma * np.min(Q[s_next_idx, :]))

        state = next_state

    print(f"Episode {ep+1}/{episodes} - Total cost: {total_cost:.3f}")

print("Training finished ✅")

# ----------------------------
# Print top actions for states where k=Kmax
# ----------------------------
for s in states:
    if s[0] == Kmax:
        best_action_idx = np.argmin(Q[state_to_idx[s], :])
        best_action = actions[best_action_idx]
        print(f"State {s}: Best action -> {best_action}, Expected cost -> {-Q[state_to_idx[s], best_action_idx]:.3f}")
