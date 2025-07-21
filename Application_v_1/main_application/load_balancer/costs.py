def is_vertical_scaling(action):
    return action in [-512, 512]

def indicator_resource_performance(a1, a2, k_next_state, u_next_state, c_next_state, Rmax, R):
    performance_metric = R * (k_next_state + a1) * u_next_state * (c_next_state + a2)
    return int(performance_metric > Rmax)


class Costs:
    @staticmethod
    def overall_cost_function(
        wadp, wperf, wres,
        k_next_state, u_next_state, c_next_state,
        action, a1, a2,
        Rmax, Kmax, R,
        c_adp=1.0, c_perf=1.0, c_res=1.0
    ):
        # Term 1: Vertical scaling indicator (normalized)
        vertical_scaling_indicator = int(is_vertical_scaling(action))
        term1 = wadp * (vertical_scaling_indicator * c_adp) / c_adp
        
        # Term 2: Performance threshold indicator (normalized)
        performance_indicator = indicator_resource_performance(a1, a2, k_next_state, u_next_state, c_next_state, Rmax, R)
        term2 = wperf * (performance_indicator * c_perf) / c_perf

        # Term 3: Resource usage cost (normalized)
        resource_usage = (k_next_state + a1) * (c_next_state + a2)
        term3 = wres * (resource_usage * c_res) / (Kmax * c_res)

        return term1 + term2 + term3

    # @staticmethod
    # def known_cost_function(wadp, wres, k_next_state, c_next_state, a1, a2, action, Kmax):
    #     # Term 1: Adaptation cost (only vertical scaling is considered here)
    #     term1 = wadp * int(is_vertical_scaling(action))
        
    #     # Term 2: Resource cost
    #     term3 = wres * (k_next_state + a1) * (c_next_state + a2) / Kmax

    #     return term1 + term3
