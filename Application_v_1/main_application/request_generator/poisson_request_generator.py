import numpy as np
import time
import requests
import os

# Endpoint and request rate setup
ENDPOINT = os.getenv("ENDPOINT", "http://195.251.56.82:8080/json_endpoint")
LAMBDA = int(os.getenv("LAMBDA", 5))  # Average requests per minute

def send_request():
    try:
        # Record the start time
        start_time = time.time()
        
        # Send the request
        response = requests.get(ENDPOINT)
        
        # Calculate the time taken for the request
        response_time = time.time() - start_time
        
        print(f"Status: {response.status_code}, Time taken: {response_time:.2f} seconds")
    
    except requests.RequestException as e:
        print(f"Request failed: {e}")

def generate_poisson_requests(rate):
    while True:
        # Generate the interval based on Poisson distribution
        interval = np.random.poisson(rate)
        time.sleep(interval / 60)  # Convert interval to minutes
        send_request()

if __name__ == "__main__":
    generate_poisson_requests(60 / LAMBDA)
