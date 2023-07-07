import requests
import threading
import time

running = True

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
            if results is not None:
                for result in results:
                    metric_value = result['value']
                has_data = check_that_metrics_have_values(metric_value)
                if has_data:
                    return metric_value[1]
                else:
                    return has_data
            else:
                return None
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
            if results is not None:
                for result in results:
                    metric_value = result['value']
                has_data = check_that_metrics_have_values(metric_value)
                if has_data:
                    return metric_value[1]
                else:
                    return has_data
            else:
                return None
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
            if results is not None:
                for result in results:
                    metric_value = result['value']
                has_data = check_that_metrics_have_values(metric_value)
                if has_data:
                    return metric_value[1]
                else:
                    return has_data
            else:
                return None
    except Exception as e:
        print("An error occured:", e)

def check_that_metrics_have_values(metrics_value):
    if metrics_value is None:
        return None
    else:
        return metrics_value

def fetch_metrics_periodically():    
    cpu_percent = get_cpu_metrics()
    ram_percent = get_memory_metrics()
    up_time = get_service_up_time()
    if cpu_percent is not None and ram_percent is not None and up_time is not None:
        print("CPU: " + cpu_percent + "% " + ("| RAM: ") + ram_percent + "% ")
    else:
        print("Metrics are not available just yet, please hold tight...")
    
    threading.Timer(10, fetch_metrics_periodically).start()
        
if __name__ == '__main__':
    print("Starting...")
    fetch_metrics_periodically()
