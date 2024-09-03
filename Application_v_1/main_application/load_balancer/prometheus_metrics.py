import requests

def get_cpu_metrics(url):
    params = {'query': 'avg(cpu_usage{job="swarm-service"})'}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data is not None and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    metric_value = results[0]['value'][1]
                    return metric_value
            return None
        else:
            return None
    except Exception as e:
        print("An error occurred during service CPU retrieval:", e)
        return None

def get_memory_metrics(url):
    params = {'query': 'avg(ram_usage{job="swarm-service"})'}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data is not None and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    metric_value = results[0]['value'][1]
                    return metric_value
            return None
        else:
            return None
    except Exception as e:
        print("An error occurred during service RAM retrieval:", e)
        return None

def get_service_up_time(url):
    params = {'query': 'sum(running_time{job="swarm-service"})'}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data is not None and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    metric_value = results[0]['value'][1]
                    return metric_value
            return None
        else:
            return None
    except Exception as e:
        print("An error occurred during service running retrieval:", e)
        return None

def get_response_time(url):
    params = {'query': 'min(rate(json_endpoint_response_time_seconds_sum[1m]) / rate(json_endpoint_response_time_seconds_count[1m]))'}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data is not None and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    metric_value = results[0]['value'][1]
                    return metric_value
            return None
        else:
            return None
    except Exception as e:
        print("An error occurred during response time retrieval:", e)
        return None
    
def get_cpu_shares(url):
    params = {'query': 'avg(cpu_shares{job="swarm-service"})'}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data is not None and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    metric_value = results[0]['value'][1]
                    return metric_value
            return None
        else:
            return None
    except Exception as e:
        print("An error occurred during response cpu share retrieval:", e)
        return None

def fetch_metrics_periodically(url):
    cpu_percent = get_cpu_metrics(url)
    ram_percent = get_memory_metrics(url)
    up_time = get_service_up_time(url)
    response_time = get_response_time(url)
    cpu_shares = get_cpu_shares(url)
    if cpu_percent is not None and ram_percent is not None and up_time is not None and response_time is not None and cpu_shares is not None:
        return cpu_percent, ram_percent, up_time, response_time, cpu_shares
    else:
        return None

def start_metrics_service(url):
    metrics = fetch_metrics_periodically(url)
    return metrics
