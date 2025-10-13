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
        print(f"Service '{service_name}' scaled in to {desired_replicas} replicas.")
        time.sleep(wait_time)
        return True
    else:
        print("Minimum replicas reached. Cannot scale in further.")
        return False

# ----------------------------
# CPU share operations
# ----------------------------
import time
import docker

def set_cpu_limit(service_name, cpu_limit, retry_attempts=5, wait_time=2):
    client = docker.from_env()
    try:
        desired_nano_cpus = int(float(cpu_limit) * 1e9)
    except Exception:
        print("[ERROR] cpu_limit must be a number (e.g. 0.5, 1.0).")
        return False

    for attempt in range(1, retry_attempts + 1):
        try:
            service = client.services.get(service_name)
            version = service.attrs['Version']['Index']
            spec = service.attrs['Spec']

            old_template = spec['TaskTemplate']

            new_resources = old_template.get('Resources', {})
            new_limits = new_resources.get('Limits', {})
            new_limits['NanoCPUs'] = desired_nano_cpus
            new_resources['Limits'] = new_limits

            new_template = old_template.copy()
            new_template['Resources'] = new_resources

            client.api.update_service(
                service.id,
                version,
                task_template=new_template,
                name=spec.get('Name'),
                labels=spec.get('Labels'),
                mode=spec.get('Mode'),
                update_config=spec.get('UpdateConfig'),
                networks=spec.get('Networks'),
                endpoint_spec=spec.get('EndpointSpec')
            )

            print(f"[OK] Updated '{service_name}' CPU limit to {cpu_limit} ({desired_nano_cpus} NanoCPUs)")
            return True

        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{retry_attempts} failed: {e}")
            time.sleep(wait_time)

    print("[ERROR] Failed to update CPU limit after retries.")
    return False

# ----------------------------
# CPU share calculation helper
# ----------------------------
def normalize_cpu_fraction(cpu_fraction):
    # Restrict CPU to common Docker fractional values
    allowed_values = [2.0, 1.0, 0.5, 0.25, 0.125]
    return cpu_fraction if cpu_fraction in allowed_values else 2.0
