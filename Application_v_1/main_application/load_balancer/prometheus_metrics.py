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

def get_response_time(url):
    # Calculate avg response time in last 30s
    promql = 'rate(json_endpoint_response_time_seconds_sum{job="swarm-service"}[30s]) \
              / rate(json_endpoint_response_time_seconds_count{job="swarm-service"}[30s])'
    
    params = {'query': promql}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data and 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    metric_value = float(results[0]['value'][1])
                    return metric_value
            return None
        else:
            print("Prometheus query failed:", response.status_code, response.text)
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
    response_time = get_response_time(url)
    cpu_shares = get_cpu_shares(url)
    return cpu_percent, response_time, cpu_shares

def start_metrics_service(url):
    cpu_percent, response_time, cpu_shares = fetch_metrics_periodically(url)
    return cpu_percent, response_time, cpu_shares

