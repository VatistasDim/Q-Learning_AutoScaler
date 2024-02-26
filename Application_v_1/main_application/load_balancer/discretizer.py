class Discretizer:
    @staticmethod
    def discretize_stack_containers_cpu_shares(cpu_shares, max_cpu_shares, num_states):
        """
        Discretizes the CPU shares for multiple containers.

        Args:
            cpu_shares (float): Mean CPU share value.
            max_cpu_shares (int): The maximum CPU shares.
            num_states (int): Number of states.

        Returns:
            int: Discretized CPU shares.
        """
        discretized_cpu_shares = min(int(cpu_shares / max_cpu_shares * num_states), num_states - 1)
        return discretized_cpu_shares
    
    @staticmethod
    def discretize_num_containers(num_containers, max_containers, num_states):
        """
        Discretizes the number of containers.

        Args:
            num_containers (int): The current number of containers.
            max_containers (int): The maximum number of containers.
            num_states (int): Number of states.

        Returns:
            int: The discretized number of containers.
        """
        discretized_value = int((num_containers / max_containers) * num_states)
        return max(0, min(discretized_value, num_states - 1))