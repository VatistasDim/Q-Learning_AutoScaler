import requests

metric_value = None

def get_cpu_metrics(url):
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
        
def get_memory_metrics(url):
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
        
def get_service_up_time(url):
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

def fetch_metrics_periodically(url):
    cpu_percent = get_cpu_metrics(url)
    ram_percent = get_memory_metrics(url)
    up_time = get_service_up_time(url)
    if cpu_percent is not None and ram_percent is not None and up_time is not None:
        #print("CPU: " + cpu_percent + "% " + ("| RAM: ") + ram_percent + "% " + ("| Service_Run_Time: ") + up_time)
        return cpu_percent, ram_percent, up_time
    else:
        return None

def start_metrics_service(running, url):
    #
    metrics = fetch_metrics_periodically(url)
    return metrics