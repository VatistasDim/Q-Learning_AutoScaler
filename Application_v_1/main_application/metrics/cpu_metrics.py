import requests
import threading
import time

def get_cpu_metrics():
    try:
        url = 'http://prometheus:9090/api/v1/query'
        params = {
            'query': "mnist_cpu_usage"
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'data' in data and 'result' in data['data']:
            results = data['data']['result']
            for result in results:
                metric_value = result['value']
        return metric_value[1]
    except Exception as e:
        print("An error occured:", e)
        
def get_memory_metrics():
    try:
        url = 'http://prometheus:9090/api/v1/query'
        params = {
            'query': "mnist_ram_usage"
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'data' in data and 'result' in data['data']:
            results = data['data']['result']
            for result in results:
                metric_value = result['value']
                # print(f'Memory Usage: {metric_value[1]}')
        return metric_value[1]
    except Exception as e:
        print("An error occured:", e)
        
def get_service_up_time():
    try:
        url = 'http://prometheus:9090/api/v1/query'
        params = {
            'query': "mnist_running_time"
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'data' in data and 'result' in data['data']:
            results = data['data']['result']
            for result in results:
                metric_value = result['value']
                # print(f'Memory Usage: {metric_value[1]}')
        return metric_value[1]
    except Exception as e:
        print("An error occured:", e)


def fetch_metrics_periodically():
    while True:
        cpu_percent = get_cpu_metrics()
        ram_percent = get_memory_metrics()
        up_time = get_service_up_time()
        print("CPU: " + cpu_percent + "% " + ("| RAM: ") + ram_percent + "% ")
        # time.sleep(15)
        
if __name__ == '__main__':
    
    fetch_thread = threading.Thread(target=fetch_metrics_periodically)
    fetch_thread.start()