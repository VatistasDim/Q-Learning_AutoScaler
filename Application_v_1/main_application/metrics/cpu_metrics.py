import requests
import threading
import time, sched

running = True
s = sched.scheduler(time.time, time.sleep)
url = 'http://prometheus:9090/api/v1/query'
metric_value = None

def get_cpu_metrics():
    params = {
            'query': "mnist_cpu_usage"
         }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data is not None and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    for result in results:
                        metric_value = result['value']
                    
                    return metric_value[1]
                
            return None
        else:
            return None
    except Exception as e:
        print("An error occurred during service uptime retrieval:", e)
        return None
        
def get_memory_metrics():
    params = {
            'query': "mnist_ram_usage"
         }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data is not None and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    for result in results:
                        metric_value = result['value']
                    
                    return metric_value[1]
                
            return None
        else:
            return None
    except Exception as e:
        print("An error occurred during service uptime retrieval:", e)
        return None
        
def get_service_up_time():
    params = {
            'query': "mnist_running_time"
         }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data is not None and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    for result in results:
                        metric_value = result['value']
                    
                    return metric_value[1]
                
            return None
        else:
            return None
    except Exception as e:
        print("An error occurred during service uptime retrieval:", e)
        return None

def fetch_metrics_periodically():
    cpu_percent = get_cpu_metrics()
    ram_percent = get_memory_metrics()
    up_time = get_service_up_time()
    if cpu_percent is not None and ram_percent is not None and up_time is not None:
        print("CPU: " + cpu_percent + "% " + ("| RAM: ") + ram_percent + "% " + ("| Service_Run_Time: ") + up_time)
    else:
        print("Metrics are not available just yet, please hold tight...")
        
while True:
    fetch_metrics_periodically()
    time.sleep(0.5)
