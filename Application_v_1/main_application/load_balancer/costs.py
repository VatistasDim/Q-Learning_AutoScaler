#the adaptation cost cadp, which accounts for the application unavailability following an adaptation.


def is_vertical_scaling(action):
    return action != 0 and action != 2

def indicator_vertical_scaling(action):
    return 1 if action == 0 else 0

def indicator_resource_performance(action1, action2, k_running_containers, u_cpu_utilization, Rmax, R):
    condition = R * ((k_running_containers + action1) * (u_cpu_utilization + action2)) > Rmax
    return 1 if condition else 0

class Costs:
    def overall_cost_function(wadp, wperf, wres, k_running_containers, action1, u_cpu_utilization, action2, Rmax, c_cpu_shares, Kmax, cres, R):
        action2_sum = sum(sum(inner_list) for inner_list in action2)
        # Term 1
        term1 = wadp * indicator_vertical_scaling(action1)
        print(f'term1:{term1}')
        # Term 2
        term2 = wperf * indicator_resource_performance(action1, action2_sum, k_running_containers, u_cpu_utilization, Rmax, R)
        print(f'term2:{term2}')
        # Term 3
        term3 = wres * (k_running_containers + action1) * (c_cpu_shares + action2_sum) / Kmax * cres
        print(f'term3:{term3}')
        # Overall cost
        cost = term1 + term2 + term3
        return cost

    def first_term_cost_function(wadp, action):
        # Term 1 (wadp * 1{vertical−scaling})
        term1 = wadp * indicator_vertical_scaling(action)
        return term1
    
    def second_term_cost_function(wperf, k_running_containers, action1, u_cpu_utilization, action2, Rmax, R):
        # Term 2 (wperf * 1{R(k+a1,u,c+a2)>Rmax})
        term2 = wperf * indicator_resource_performance(action1, action2, k_running_containers, u_cpu_utilization, Rmax, R)
        return term2
    
    def third_term_cost_function(wres, k_running_containers, a1, c_cpu_shares, a2, Kmax, cres):
        # Term 3 (wres * (k + a1)(c + a2) / Kmax * cres)
        term3 = wres * (k_running_containers + a1) * (c_cpu_shares + a2) / Kmax * cres
        return term3
    
    # https://en.wikipedia.org/wiki/Indicator_function
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
    

