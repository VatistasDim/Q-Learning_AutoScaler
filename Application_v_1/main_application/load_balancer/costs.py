def is_vertical_scaling(action):
    return action in [-512, 512]

def indicator_resource_performance(a1, a2, k_next_state, u_next_state, c_next_state, Rmax, R):
    performance_metric = R * (k_next_state + a1) * u_next_state * (c_next_state + a2)
    return int(performance_metric > Rmax)


class Costs:
    @staticmethod
    def overall_cost_function(wadp, wperf, wres, k_next_state, u_next_state, c_next_state, action, a1, a2, Rmax, Kmax, R):
        # Term 1: Weight for vertical scaling
        term1 = wadp * int(is_vertical_scaling(action))
        
        # Term 2: Weight for resource performance
        term2 = wperf * indicator_resource_performance(a1, a2, k_next_state, u_next_state, c_next_state, Rmax, R)
        
        # Term 3: Resource cost
        term3 = wres * (k_next_state + a1) * (c_next_state + a2) / Kmax
        
        # Overall cost
        return term1 + term2 + term3
