import prometheus_metrics as prometheus_metrics

def fetch_metrics_periodically(url):
    cpu_percent = prometheus_metrics.get_cpu_metrics(url)
    ram_percent = prometheus_metrics.get_memory_metrics(url)
    up_time = prometheus_metrics.get_service_up_time(url)
    if cpu_percent is not None and ram_percent is not None and up_time is not None:
        return cpu_percent, ram_percent, up_time
    else:
        return None

def start_metrics_service(running, url):
    metrics = fetch_metrics_periodically(url)
    return metrics