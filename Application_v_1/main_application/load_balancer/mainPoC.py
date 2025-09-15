import numpy as np
import random
import os
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
import prometheus_metrics

learning_report = {
    "episodes": [],
    "best_actions_per_state": {},
    "worst_actions_per_state": {},
    "overall_statistics": {}
}

# ----------------------------
# Parameters
# ----------------------------
Kmax = 10        # maximum containers
u_max = 100      # CPU utilization (%)
c_max = 100      # CPU shares
u_quantum = 10
c_quantum = 10
STEP_DURATION = 10  # seconds

docker_api = DockerAPI(stack_name="mystack_application")

# Q-learning parameters
episodes = 100
steps_per_episode = 10
alpha = 0.1
gamma = 0.95
epsilon = 0.1

# Scaling & cost parameters
Rmax = 80  # max acceptable response time
w_adp = 0.33
w_perf = 0.33
w_res = 0.33

# Prometheus URL
PROM_URL = "http://prometheus:9090/api/v1/query"

# ----------------------------
# Helper functions
# ----------------------------
def discretize(value, quantum, v_min, v_max):
    value = max(v_min, min(value, v_max))
    return int(round(value / quantum) * quantum)

def fetch_data():
    max_attempts = 100
    for attempt in range(max_attempts):
        try:
            cpu_percent, ram_percent, time_up, response_time, cpu_shares = prometheus_metrics.start_metrics_service(url=PROM_URL)
            if time_up != '0':
                cpu_percent = int(float(cpu_percent))
                ram_percent = int(float(ram_percent))
                time_up = int(float(time_up))
                response_time = float(response_time)
                cpu_shares = calculate_cpu_shares(get_current_cpu_shares("mystack_application"))
                if None in (cpu_percent, ram_percent, time_up, response_time, cpu_shares):
                    continue
                return cpu_percent, ram_percent, time_up, response_time, cpu_shares
        except Exception as e:
            print(f"Error: An error occurred during service metrics retrieval (Attempt {attempt + 1}/{max_attempts}):", e)
            if attempt < max_attempts - 1:
                time.sleep(5)
    print("Failed to retrieve metrics after multiple attempts.")
    return None, None, None, None, None

def apply_action(service_name, state, action, prometheus_url=None):
    k, u, c = state

    # --- Execute scaling ---
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

    # --- Get metrics from Prometheus ---
    cpu_percent, ram_percent, time_up, response_time, cpu_shares = fetch_data(service_name, prometheus_url)

    # Fallback if Prometheus is missing something
    if response_time is None:
        response_time = u  # just keep previous response time if metric missing
    if cpu_shares is None:
        cpu_shares = c     # keep previous cpu_shares

    # Update state (replicas and cpu_shares from Docker)
    k_actual = get_current_replica_count(service_name) or 1
    c_actual = cpu_shares or get_current_cpu_shares(service_name) or c

    # New discrete state
    next_state = (
        min(max(1, k_actual), Kmax),
        discretize(response_time, u_quantum, 0, u_max),
        discretize(c_actual, c_quantum, c_quantum, c_max)
    )

    return next_state, response_time

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
# Create a mapping from each state tuple (k, u, c) to a unique index in the states list.
state_to_idx = {s: i for i, s in enumerate(states)}

# ----------------------------
# Actions
# ----------------------------
actions = [("vscale", -10), ("hscale", -1), ("noop", 0), ("hscale", +1), ("vscale", +10)]
n_actions = len(actions)

# ----------------------------
# Q-learning loop
# ----------------------------
Q = np.zeros((len(states), n_actions))

time.sleep(30)

for ep in range(episodes):
    # Initial state
    state = (
        min(max(1, get_current_replica_count("mystack_application")), Kmax),
        random.randint(20, 80),
        min(max(c_quantum, get_current_cpu_shares("mystack_application")), c_max)
    )

    total_cost = 0
    total_k = 0
    total_c = 0
    performance_met = 0
    scaling_steps = 0

    for step in range(steps_per_episode):
        step_start = time.time() # start timer
        s_idx = state_to_idx.get(state)
        if s_idx is None:
            # safety check
            state = (min(max(1, state[0]), Kmax),
                     discretize(state[1], u_quantum, 0, u_max),
                     discretize(state[2], c_quantum, c_quantum, c_max))
            s_idx = state_to_idx[state]

        # Epsilon-greedy
        a_idx = np.random.randint(n_actions) if random.random() < epsilon else np.argmin(Q[s_idx, :])
        action = actions[a_idx]

        # Count scaling actions
        if action[0] in ["hscale", "vscale"] and action[1] != 0:
            scaling_steps += 1

        next_state, R_current = apply_action("mystack_application", state, action, prometheus_url=PROM_URL)
        if R_current <= Rmax:
            performance_met += 1

        # Accumulate resource usage
        total_k += next_state[0]
        total_c += next_state[2]

        # Extract action effect for cost function
        a1 = action[1] if action[0] == "hscale" else 0
        a2 = action[1] if action[0] == "vscale" else 0

        cost = Costs.overall_cost_function(
            wadp=w_adp, wperf=w_perf, wres=w_res,
            k_next_state=next_state[0],
            u_next_state=next_state[1],
            c_next_state=next_state[2],
            action=action,
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
        
        # --- Wait until STEP_DURATION min has passed ---
        elapsed = time.time() - step_start
        if elapsed < STEP_DURATION:
            time.sleep(STEP_DURATION - elapsed)
       
        # Print step info
        print(f"Episode {ep+1}, Step {step+1}/{steps_per_episode}, Action: {action}, State: {state}, R_current: {R_current:.2f}")
    
    best_actions = {}
    worst_actions = {}
    for s in states:
        s_idx = state_to_idx[s]
        best_idx = np.argmin(Q[s_idx, :])
        worst_idx = np.argmax(Q[s_idx, :])
        best_actions[s] = (actions[best_idx], -Q[s_idx, best_idx])
        worst_actions[s] = (actions[worst_idx], -Q[s_idx, worst_idx])

    episode_stats = {
        "episode": ep + 1,
        "total_cost": total_cost,
        "performance_met_percentage": performance_met / steps_per_episode * 100,
        "scaling_frequency_percentage": scaling_steps / steps_per_episode * 100,
        "avg_replicas": total_k / steps_per_episode,
        "avg_cpu_shares": total_c / steps_per_episode,
        "best_actions": best_actions,
        "worst_actions": worst_actions
    }
    learning_report["episodes"].append(episode_stats)

    # Episode summary
    avg_k = total_k / steps_per_episode
    avg_c = total_c / steps_per_episode
    print(f"\nEpisode {ep+1} Summary:")
    print(f"  Total cost: {total_cost:.3f}")
    print(f"  Performance goal met: {performance_met}/{steps_per_episode} steps ({performance_met/steps_per_episode*100:.1f}%)")
    print(f"  Average replicas / CPU shares: {avg_k:.2f} / {avg_c:.2f}")
    print(f"  Scaling frequency: {scaling_steps}/{steps_per_episode} steps ({scaling_steps/steps_per_episode*100:.1f}%)")
    print("-"*50)

print("Training finished..")

# ----------------------------
# Print top actions for states where k=Kmax
# ----------------------------
for s in states:
    if s[0] == Kmax:
        best_action_idx = np.argmin(Q[state_to_idx[s], :])
        best_action = actions[best_action_idx]
        print(f"State {s}: Best action -> {best_action}, Expected cost -> {-Q[state_to_idx[s], best_action_idx]:.3f}")

# Overall summary
total_episodes = len(learning_report["episodes"])
avg_cost = sum(ep["total_cost"] for ep in learning_report["episodes"]) / total_episodes
avg_perf = sum(ep["performance_met_percentage"] for ep in learning_report["episodes"]) / total_episodes
avg_scaling = sum(ep["scaling_frequency_percentage"] for ep in learning_report["episodes"]) / total_episodes

learning_report["overall_statistics"] = {
    "avg_total_cost": avg_cost,
    "avg_performance_met_percentage": avg_perf,
    "avg_scaling_frequency_percentage": avg_scaling
}

# Save to file
report_file = "/logs/q-learning-final-log.txt"
with open(report_file, "w") as f:
    f.write("Q-Learning Final Report\n")
    f.write("="*50 + "\n\n")
    
    for ep_stat in learning_report["episodes"]:
        f.write(f"Episode {ep_stat['episode']}\n")
        f.write(f"  Total cost: {ep_stat['total_cost']:.2f}\n")
        f.write(f"  Performance goal met: {ep_stat['performance_met_percentage']:.1f}%\n")
        f.write(f"  Scaling frequency: {ep_stat['scaling_frequency_percentage']:.1f}%\n")
        f.write(f"  Average replicas: {ep_stat['avg_replicas']:.2f}\n")
        f.write(f"  Average CPU shares: {ep_stat['avg_cpu_shares']:.2f}\n")
        f.write("  Best actions per state:\n")
        for state, (action, cost) in ep_stat['best_actions'].items():
            f.write(f"    State {state}: Action {action}, Expected cost {cost:.2f}\n")
        f.write("  Worst actions per state:\n")
        for state, (action, cost) in ep_stat['worst_actions'].items():
            f.write(f"    State {state}: Action {action}, Expected cost {cost:.2f}\n")
        f.write("\n")
    
    f.write("="*50 + "\n")
    f.write("Overall statistics:\n")
    for key, val in learning_report["overall_statistics"].items():
        f.write(f"  {key}: {val:.2f}\n")

print(f"Final report saved to {report_file}")
