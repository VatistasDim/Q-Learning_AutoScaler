import numpy as np
import random
import os
import time
import json
from scalingOperations import (
    scale_out,
    scale_in,
    set_cpu_limit,
    normalize_cpu_fraction,
    get_current_replica_count,
    get_current_cpu_shares
)
from costs import Costs
from docker_api import DockerAPI
import prometheus_metrics

# ----------------------------
# Learning report
# ----------------------------
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
episodes = 2
steps_per_episode = 3
alpha = 0.1
gamma = 0.95
epsilon = 0.1

# Scaling & cost parameters
Rmax = 0.80  # max acceptable response time
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
    return int(round(value / quantum) * quantum) # Check if goes up after rounding. remove / Quantum and * quantum.

def fetch_data(service_name="mystack_application", max_attempts=30, retry_delay=5):
    for attempt in range(max_attempts):
        try:
            cpu_percent, response_time, cpu_shares = prometheus_metrics.start_metrics_service(PROM_URL)

            if None in (cpu_percent, response_time, cpu_shares) and attempt < max_attempts - 1:
                time.sleep(retry_delay)
                continue

            cpu_percent = int(float(cpu_percent))
            response_time = float(response_time)
            cpu_shares = calculate_cpu_shares(get_current_cpu_shares(service_name))

            return cpu_percent, response_time, cpu_shares

        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(retry_delay)

    print("[fetch_data] Failed to retrieve metrics after multiple attempts.")
    return None, None, None

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
        set_cpu_limit(service_name, normalize_cpu_fraction(new_cpu / 10))
    # noop does nothing
    
    time.sleep(30)

    # --- Get metrics ---
    cpu_percent, response_time, cpu_shares = fetch_data()

    if response_time is None:
        response_time = u
    if cpu_shares is None:
        cpu_shares = c

    k_actual = get_current_replica_count(service_name) or 1
    c_actual = cpu_shares or get_current_cpu_shares(service_name) or c

    next_state = (
        min(max(1, k_actual), Kmax),
        discretize(cpu_percent, u_quantum, 0, u_max),
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
state_to_idx = {s: i for i, s in enumerate(states)}

# ----------------------------
# Actions
# ----------------------------
actions = [("vscale", -10), ("hscale", -1), ("noop", 0), ("hscale", +1), ("vscale", +10)]
n_actions = len(actions)

Q = np.zeros((len(states), n_actions))

log_file = "/logs/q-learning-steps.txt"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

with open(log_file, "w") as lf:
    lf.write("=== Q-Learning Training Log ===\n\n")

# ----------------------------
# Training loop
# ----------------------------
for ep in range(episodes):
    k0 = min(max(1, get_current_replica_count("mystack_application")), Kmax)
    u0 = random.randint(20, 80)
    c0 = min(max(c_quantum, get_current_cpu_shares("mystack_application")), c_max)
    state = (k0, u0, c0)

    total_cost = 0
    total_k = 0
    total_c = 0
    performance_met = 0
    scaling_steps = 0
    step_logs = []

    for step in range(steps_per_episode):
        
        step_start = time.time()
        k, u, c = state
        state = (
            min(max(1, k), Kmax),
            discretize(u, u_quantum, 0, u_max),
            discretize(c, c_quantum, c_quantum, c_max)
        )
        s_idx = state_to_idx[state]

        a_idx = np.random.randint(n_actions) if random.random() < epsilon else np.argmin(Q[s_idx, :])
        action = actions[a_idx]

        a1 = action[1] if action[0] == "hscale" else 0
        a2 = action[1] if action[0] == "vscale" else 0

        next_state, R_next = apply_action("mystack_application", state, action, prometheus_url=PROM_URL)

        costs = Costs.overall_cost_function(
            wadp=w_adp, wperf=w_perf, wres=w_res,
            k_next_state=next_state[0],
            u_next_state=next_state[1],
            c_next_state=next_state[2],
            action=action, a1=a1, a2=a2,
            Rmax=Rmax, Kmax=Kmax,
            response_time=R_next
        )

        total_cost = costs["total"]
        total_k += next_state[0]
        total_c += next_state[2]
        if R_next <= Rmax:
            performance_met += 1
        if action[0] in ["hscale", "vscale"] and action[1] != 0:
            scaling_steps += 1

        # Q-update
        s_next_idx = state_to_idx[next_state]
        Q[s_idx, a_idx] = (1 - alpha) * Q[s_idx, a_idx] + alpha * (costs["total"] + gamma * np.min(Q[s_next_idx, :]))
        state = next_state

        # Log step
        step_log = (
            f"[Episode {ep+1}, Step {step+1}] "
            f"State={state}, Action={action}, Response={R_next:.2f}, "
            f"Cost={costs['total']:.4f} "
            f"(Adapt={costs['term1']:.4f}, Perf={costs['term2']:.4f}, Res={costs['term3']:.4f})")
        print(step_log)
        with open(log_file, "a") as lf:
            lf.write(step_log + "\n")

        step_logs.append({
            "step": step + 1,
            "state": state,
            "action": action,
            "R_next": R_next,
            "cost": costs["total"],
            "term1_adaptation": costs["term1"],
            "term2_performance": costs["term2"],
            "term3_resources": costs["term3"]
        })

        elapsed = time.time() - step_start
        if elapsed < STEP_DURATION:
            time.sleep(STEP_DURATION - elapsed)

    # --- Episode summary ---
    avg_k = total_k / steps_per_episode
    avg_c = total_c / steps_per_episode
    summary = (
        f"--- Episode {ep+1} Summary ---\n"
        f"Total Cost: {total_cost:.4f}\n"
        f"Performance Met: {performance_met / steps_per_episode * 100:.2f}%\n"
        f"Scaling Frequency: {scaling_steps / steps_per_episode * 100:.2f}%\n"
        f"Average Replicas: {avg_k:.2f}, Average CPU: {avg_c:.2f}\n"
        "------------------------------\n\n"
    )
    print(summary)
    with open(log_file, "a") as lf:
        lf.write(summary)

# --- Best/Worst actions per state ---
# for s in states:
#     s_idx = state_to_idx[s]
#     best_idx = np.argmin(Q[s_idx, :])
#     worst_idx = np.argmax(Q[s_idx, :])
#     learning_report["best_actions_per_state"][s] = {"action": actions[best_idx], "cost": Q[s_idx, best_idx]}
#     learning_report["worst_actions_per_state"][s] = {"action": actions[worst_idx], "cost": Q[s_idx, worst_idx]}

# # --- Overall statistics ---
# total_episodes = len(learning_report["episodes"])
# learning_report["overall_statistics"] = {
#     "avg_total_cost": np.mean([ep["total_cost"] for ep in learning_report["episodes"]]),
#     "avg_performance_met_percentage": np.mean([ep["performance_met_percentage"] for ep in learning_report["episodes"]]),
#     "avg_scaling_frequency_percentage": np.mean([ep["scaling_frequency_percentage"] for ep in learning_report["episodes"]])
# }

# # --- Save JSON report ---
# report_file = "/logs/q-learning-detailed.json"
# os.makedirs(os.path.dirname(report_file), exist_ok=True)
# with open(report_file, "w") as f:
#     json.dump(learning_report, f, indent=2)

# print(f"Training finished. Full report saved to {report_file}")
