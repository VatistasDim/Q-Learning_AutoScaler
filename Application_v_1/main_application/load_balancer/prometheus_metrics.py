import requests

def get_cpu_metrics(url):
    params = {'query': "cpu_usage"}
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
    params = {'query': "ram_usage"}
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
    params = {'query': "running_time"}
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
    params = {'query': "response_time"}
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
    params = {'query': "cpu_shares"}
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
        print(f'ERROR: There was an issue during the fetch of metrics. \n Relative logs may help. cpu_percent = {cpu_percent}, ram_percent = {ram_percent}, up_time = {up_time}, response_time = {response_time}, cpu_shares = {cpu_shares}')
        return None

def start_metrics_service(url):
    metrics = fetch_metrics_periodically(url)
    return metrics
