import docker

class DockerAPI:
    def __init__(self, stack_name):
        self.client = docker.from_env()
        self.stack_name = stack_name

    def get_running_containers_by_filters(self, filter):
        running_containers = self.client.containers.list(filters=filter)
        return running_containers

    def get_cpu_shares_from_container(self, container):
        stats = container.stats(stream=False)
        cpu_stats = stats['cpu_stats']
        cpu_shares = cpu_stats['cpu_usage']['percpu_usage']
        return cpu_shares

    def get_stack_containers_cpu_shares(self, service_name):
        filter = {'status': 'running'}
        running_containers = self.get_running_containers_by_filters(filter)

        # Filter containers based on name
        load_balancer_containers = [container for container in running_containers if container.name.startswith(service_name)]

        cpu_shares_dict = {}
        for container in load_balancer_containers:
            cpu_shares = self.get_cpu_shares_from_container(container)
            cpu_shares_dict[container.id] = cpu_shares
        return cpu_shares_dict
