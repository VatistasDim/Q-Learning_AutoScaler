def is_vertical_scaling(action):
    return action[0] == "vscale" and action[1] != 0

class Costs:
    @staticmethod
    def overall_cost_function(
        wadp, wperf, wres,
        k_next_state, u_next_state, c_next_state,
        action, a1, a2,
        Rmax, Kmax, response_time
    ):
        # --- Term 1: Adaptation cost (vertical scaling) ---
        vertical_scaling_indicator = int(is_vertical_scaling(action))
        term1 = wadp * vertical_scaling_indicator

        # --- Term 2: Performance penalty ---
        k_effective = k_next_state + a1
        c_effective = c_next_state + a2
        performance_violation = int(response_time > Rmax)
        term2 = wperf * performance_violation

        # --- Term 3: Resource usage cost ---
        resource_usage = k_effective * c_effective
        term3 = wres * (resource_usage / max(1, Kmax * c_effective))  # normalize by max possible

        # --- Total cost ---
        total_cost = term1 + term2 + term3
        return total_cost, term1, term2, term3
