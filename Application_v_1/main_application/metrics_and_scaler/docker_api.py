import docker

def scale_service(service_name, desired_replicas):
    client = docker.from_env()
    service = client.services.get(service_name)
    service.scale(desired_replicas)
    print(f"Service '{service_name}' scaled to {desired_replicas} replicas.")
    
def get_current_replica_count(service_name):
    client = docker.from_env()
    try:
        service = client.services.get(service_name)
        return service.attrs['Spec']['Mode']['Replicated']['Replicas']
    except docker.errors.NotFound:
        return None
