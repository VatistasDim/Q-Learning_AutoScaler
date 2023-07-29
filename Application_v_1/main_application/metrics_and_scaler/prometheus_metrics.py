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
        print("An error occurred during service CPU retrieval:", e)
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
        print("An error occurred during service RAM retrieval:", e)
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
        print("An error occurred during service running retrieval:", e)
        return None
