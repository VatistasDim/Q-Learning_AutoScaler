import random

class DockerAPI:
    def __init__(self, stack_name):
        self.stack_name = stack_name

    def get_stack_state(self, service_name):
        """
        Mock state provider:
        Returns (k, u, c) where
        k = number of containers
        u = utilization %
        c = cpu shares
        """
        # pretend your service has between 1–10 containers
        k = random.randint(1, 10)

        # utilization fluctuates between 0–100%
        u = random.randint(0, 100)

        # cpu shares between 10–100 (step of 10)
        c = random.choice(range(10, 110, 10))

        return (k, u, c)

    def get_running_containers_by_filters(self, filter):
        # Mocked empty
        return []

    def get_cpu_shares_from_container(self, container):
        # Mock: return random cpu shares per "cpu core"
        return [random.randint(10000000, 20000000) for _ in range(4)]

    def get_stack_containers_cpu_shares(self, service_name):
        # Mock: pretend you have 2 containers with cpu shares
        return {
            "container1": [random.randint(10000000, 20000000) for _ in range(4)],
            "container2": [random.randint(10000000, 20000000) for _ in range(4)],
        }
