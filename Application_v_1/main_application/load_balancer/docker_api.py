import docker

class DockerAPI:
    def __init__(self, stack_name):
        self.client = docker.from_env()
        self.stack_name = stack_name

    def get_running_containers_by_filters(self, filter):
        running_containers = self.client.containers.list(filters=filter)
        return running_containers

    def get_cpu_percent(self, container):
        stats = container.stats(stream=False)
        cpu_stats = stats["cpu_stats"]
        precpu_stats = stats["precpu_stats"]

        # Protection against division by zero
        cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - precpu_stats["cpu_usage"]["total_usage"]
        system_delta = cpu_stats["system_cpu_usage"] - precpu_stats["system_cpu_usage"]

        cpu_percent = 0.0
        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * len(cpu_stats["cpu_usage"]["percpu_usage"]) * 100.0

        return cpu_percent

    def get_stack_state(self, service_name):
        filter = {"status": "running"}
        running_containers = self.get_running_containers_by_filters(filter)

        load_balancer_containers = [
            container for container in running_containers if container.name.startswith(service_name)
        ]

        cpu_usages = [self.get_cpu_percent(container) for container in load_balancer_containers]

        if not cpu_usages:
            return (0, 0.0, 0.0)  # no containers running

        num_containers = len(cpu_usages)
        avg_cpu = sum(cpu_usages) / num_containers
        max_cpu = max(cpu_usages)

        return (num_containers, round(avg_cpu, 2), round(max_cpu, 2))
