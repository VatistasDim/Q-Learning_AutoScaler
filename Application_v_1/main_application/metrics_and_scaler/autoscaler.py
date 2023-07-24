import service_metrics as service_metrics, docker_api, logging, time

url = 'http://prometheus:9090/api/v1/query'
cpu_threshold = 20
ram_threshold = 15
service_name = 'mystack_mnist'

max_replicas = 7
min_replicas = 1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_data():
    metrics = service_metrics.start_metrics_service(True, url)
    if metrics is not None:
        time_up = metrics[2]
        if time_up != '0':
            cpu_percent = int(float(metrics[0]))
            ram_percent = int(float(metrics[1]))
            time_up = int(float(time_up))
            return cpu_percent, ram_percent, time_up
    else:
        print("No metrics available, wait...")
        
def get_current_replica_count(service_name):
    return docker_api.get_current_replica_count(service_name)

def scale(cpu_value, ram_value):
    if cpu_value is not None and ram_value is not None and cpu_value > cpu_threshold and ram_value > ram_threshold:
        current_replicas = get_current_replica_count(service_name)
        if current_replicas < max_replicas:
            desired_replicas = min(current_replicas + 1, max_replicas)
            docker_api.scale_service(service_name, desired_replicas)
            logger.info(f"Current Replicas:{current_replicas}")
            logger.info(f"Horizontal Scale Up: Replicas increased to {desired_replicas}")
        else:
            logger.info(f"Already at maximum replicas: {max_replicas}")
    elif cpu_value is not None and ram_value is not None and cpu_value <= cpu_threshold and ram_value <= ram_threshold:
        current_replicas = get_current_replica_count(service_name)
        if current_replicas > min_replicas:
            desired_replicas = max(current_replicas - 1, min_replicas)
            docker_api.scale_service(service_name, desired_replicas)
            logger.info(f"Current Replicas:{current_replicas}")
            logger.info(f"Horizontal Scale Down: Replicas decreased to {desired_replicas}")
        else:
            logger.info(f"Already at minimum replicas: {min_replicas}")    
    
def main():
    while True:
        tuple_values = fetch_data()
        if tuple_values is not None:
            scale(tuple_values[1], tuple_values[2])
        time.sleep(15)

if __name__ == "__main__":
    main()