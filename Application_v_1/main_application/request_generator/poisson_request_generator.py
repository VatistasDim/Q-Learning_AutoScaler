import numpy as np
import time
import requests

# Define the endpoint and Poisson rate (λ)
ENDPOINT = "http://your-server-endpoint/api"
LAMBDA = 5  # Average requests per minute (adjust as needed)

def send_request():
    try:
        response = requests.get(ENDPOINT)
        print(f"Status: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")

def generate_poisson_requests(rate):
    while True:
        # Wait for a random interval based on Poisson distribution
        interval = np.random.poisson(rate)
        time.sleep(interval / 60)  # Convert interval to minutes
        send_request()

if __name__ == "__main__":
    generate_poisson_requests(60 / LAMBDA)
