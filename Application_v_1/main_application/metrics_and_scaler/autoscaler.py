from cpu_metrics import start_metrics_service
from docker_env import scale_service

url = 'http://prometheus:9090/api/v1/query'
cpu_threshold = 20
ram_threshold = 15
service_name = 'mystack_mnist'

def fetch_data():
    metrics = start_metrics_service(True, url)
    if metrics is not None:
        time_up = metrics[2]
        if time_up != '0':
            cpu_percent = int(float(metrics[0]))
            ram_percent = int(float(metrics[1]))
            time_up = int(float(time_up))
            #print(cpu_percent, ram_percent, time_up)
            return cpu_percent, ram_percent, time_up
    else:
        print("No metrics available, wait...")
    
def scale(cpu_value, ram_value):
    if cpu_value is not None and ram_value is not None and cpu_value > cpu_threshold and ram_value > ram_threshold:
        scale_service(service_name, 1)
        print("Horizontal_Scale_up")
    
def main():
    while True:
        tuple_values = fetch_data()
        if tuple_values is not None:
            scale(tuple_values[1], tuple_values[2])

if __name__ == "__main__":
    main()