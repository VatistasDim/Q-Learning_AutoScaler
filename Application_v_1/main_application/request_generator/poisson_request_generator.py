import numpy as np
import time
import requests
import os
from datetime import datetime

# Endpoint and request rate setup
ENDPOINT = os.getenv("ENDPOINT", "http://195.251.56.82:8080/json_endpoint")
LAMBDA = int(os.getenv("LAMBDA", 5))  # Average requests per minute
INTERVAL = 60 / LAMBDA  # mean time between requests (in seconds)

def send_request():
    try:
        start_time = time.time()
        response = requests.get(ENDPOINT)
        duration = time.time() - start_time

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Status: {response.status_code}, Time taken: {duration:.2f}s")

    except requests.RequestException as e:
        print(f"[{datetime.now()}] Request failed: {e}")

def generate_poisson_requests(rate_seconds):
    while True:
        interval = np.random.exponential(rate_seconds)
        time.sleep(interval)
        send_request()

if __name__ == "__main__":
    print(f"Starting request generator to {ENDPOINT} at ~{LAMBDA} requests/min...")
    generate_poisson_requests(INTERVAL)