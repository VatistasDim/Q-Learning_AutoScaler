import get_service_api

def fetch_metrics_periodically(url):
    cpu_percent = get_service_api.get_cpu_metrics(url)
    ram_percent = get_service_api.get_memory_metrics(url)
    up_time = get_service_api.get_service_up_time(url)
    if cpu_percent is not None and ram_percent is not None and up_time is not None:
        return cpu_percent, ram_percent, up_time
    else:
        return None

def start_metrics_service(running, url):
    metrics = fetch_metrics_periodically(url)
    return metrics