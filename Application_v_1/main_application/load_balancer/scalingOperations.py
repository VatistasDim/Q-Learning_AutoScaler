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
    """
    Set CPU limit for a swarm service.
    cpu_limit: float (π.χ. 0.5, 1.0, 2.0) => μετατρέπεται σε NanoCPUs (int).
    Επιστρέφει True αν επιτύχει, False αλλιώς.
    """
    client = docker.from_env()
    try:
        desired_nano_cpus = int(float(cpu_limit) * 1e9)
    except Exception:
        print("[ERROR] cpu_limit must be a number (e.g. 0.5, 1.0).")
        return False

    for attempt in range(1, retry_attempts + 1):
        try:
            service = client.services.get(service_name)
            spec = service.attrs.get('Spec', {})
            task_template = spec.get('TaskTemplate', {})

            # Ensure Resources / Limits dicts υφίστανται
            resources = task_template.get('Resources') or {}
            limits = resources.get('Limits') or {}
            limits['NanoCPUs'] = desired_nano_cpus
            resources['Limits'] = limits
            task_template['Resources'] = resources
            spec['TaskTemplate'] = task_template

            # Πάρε την τρέχουσα version index — απαιτείται από το update
            version = service.attrs.get('Version', {}).get('Index')
            if version is None:
                raise RuntimeError("Could not read service Version.Index")

            # Χρησιμοποιούμε το low-level API update_service με version ως δεύτερο arg
            client.api.update_service(
                service.id,
                version,
                task_template=task_template,
                name=spec.get('Name'),
                labels=spec.get('Labels'),
                mode=spec.get('Mode'),
                update_config=spec.get('UpdateConfig'),
                networks=spec.get('Networks'),
                endpoint_spec=spec.get('EndpointSpec')
            )

            print("[OK] Set {} CPUs ({} NanoCPUs) for '{}'".format(cpu_limit, desired_nano_cpus, service_name))
            time.sleep(wait_time)
            return True

        except docker.errors.NotFound:
            print("[ERROR] Service '{}' not found.".format(service_name))
            return False
        except docker.errors.APIError as e:
            print(f"[WARN] Docker API error on attempt {attempt}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[WARN] Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    print("[FAIL] Could not set CPU limit for '{}' after {} attempts.".format(service_name, retry_attempts))
    return False

# ----------------------------
# CPU share calculation helper
# ----------------------------
def normalize_cpu_fraction(cpu_fraction):
    # Restrict CPU to common Docker fractional values
    allowed_values = [2.0, 1.0, 0.5, 0.25, 0.125]
    return cpu_fraction if cpu_fraction in allowed_values else 2.0
