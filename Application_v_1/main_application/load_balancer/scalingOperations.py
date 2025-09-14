import docker
import time

# ----------------------------
# Configurable parameters
# ----------------------------
max_replicas = 10
wait_time = 2  # seconds to wait after scaling operations

# ----------------------------
# Docker Utilities
# ----------------------------
def get_docker_client():
    return docker.from_env()

def get_current_replica_count(service_name):
    client = get_docker_client()
    try:
        service = client.services.get(service_name)
        return service.attrs['Spec']['Mode']['Replicated']['Replicas']
    except docker.errors.NotFound:
        print(f"Error: Service '{service_name}' not found")
        return None

def get_current_cpu_shares(service_name):
    client = get_docker_client()
    try:
        service = client.services.get(service_name)
        resources = service.attrs['Spec']['TaskTemplate'].get('Resources', {})
        cpu_shares = resources.get('Limits', {}).get('NanoCPUs', 0) / 1_000_000_000
        return cpu_shares
    except docker.errors.NotFound:
        print(f"Error: Service '{service_name}' not found")
        return 0

# ----------------------------
# Scaling operations
# ----------------------------
def scale_out(service_name, scale_factor=1):
    client = get_docker_client()
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is None:
        return False

    desired_replicas = current_replicas + scale_factor
    if desired_replicas <= max_replicas:
        service = client.services.get(service_name)
        service.scale(desired_replicas)
        print(f"Log: Service '{service_name}' scaled out to {desired_replicas} replicas.")
        time.sleep(wait_time)
        return True
    else:
        print("Log: Maximum replicas reached. Cannot scale out further.")
        return False

def scale_in(service_name, scale_factor=1):
    client = get_docker_client()
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is None:
        return False

    if current_replicas > 1:
        desired_replicas = current_replicas - scale_factor
        service = client.services.get(service_name)
        service.scale(desired_replicas)
        print(f"Log: Service '{service_name}' scaled in to {desired_replicas} replicas.")
        time.sleep(wait_time)
        return True
    else:
        print("Log: Minimum replicas reached. Cannot scale in further.")
        return False

# ----------------------------
# CPU share operations
# ----------------------------
def set_cpu_shares(service_name, cpu_shares, retry_attempts=5):
    client = get_docker_client()
    for attempt in range(retry_attempts):
        try:
            service = client.services.get(service_name)
            resources = service.attrs['Spec']['TaskTemplate'].get('Resources', {})
            if 'Limits' not in resources:
                resources['Limits'] = {}

            current_shares = resources['Limits'].get('NanoCPUs', 0)
            desired_shares_nano = int(cpu_shares * 1_000_000_000)

            resources['Limits']['NanoCPUs'] = desired_shares_nano
            service.update(resources=resources)
            print(f"Log: CPU shares set to {cpu_shares} ({desired_shares_nano} NanoCPUs) for service '{service_name}'")
            time.sleep(wait_time)
            return True
        except docker.errors.NotFound:
            print(f"Error: Service '{service_name}' not found.")
            break
        except Exception as e:
            print(f"Warning: Attempt {attempt+1} failed with error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    print(f"Error: Failed to set CPU shares for '{service_name}' after {retry_attempts} attempts.")
    return False

# ----------------------------
# CPU share calculation helper
# ----------------------------
def calculate_cpu_shares(cpu_fraction):
    # Restrict CPU to common Docker fractional values
    allowed_values = [2.0, 1.0, 0.5, 0.25, 0.125]
    return cpu_fraction if cpu_fraction in allowed_values else 2.0
