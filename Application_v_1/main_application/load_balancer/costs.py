def is_vertical_scaling(action):
    return action != 0 and action != 2

class Costs:
    def calculate_adaptation_cost(w_adp, action):
        # Check if the action involves vertical scaling
        return w_adp if is_vertical_scaling(action) else 0

    def calculate_performance_penalty(response_time, w_perf, R_max):
        # Check if the response time exceeds R_max
        return w_perf if response_time > R_max else 0

    def calculate_resource_cost(w_res, num_containers, cpu_shares, max_replicas, c_res):
        k_a1_term = num_containers
        c_a2_term = cpu_shares
        resource_cost = w_res * (k_a1_term) * (c_a2_term) * c_res / max_replicas
        return resource_cost
    

