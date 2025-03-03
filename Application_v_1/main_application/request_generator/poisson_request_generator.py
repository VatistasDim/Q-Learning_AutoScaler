import numpy as np
import time
import requests
import os

# Endpoint and request rate setup
ENDPOINT = os.getenv("ENDPOINT", "http://195.251.56.82:8080/json_endpoint")
LAMBDA = int(os.getenv("LAMBDA", 20000))  # Average requests per minute

total_requests = 0
successful_requests = 0
failed_requests = 0
total_response_time = 0

def send_request():
    global total_requests, successful_requests, failed_requests, total_response_time
    
    try:
        # Record the start time
        start_time = time.time()
        
        # Send the request
        response = requests.get(ENDPOINT)
        
        # Calculate the time taken for the request
        response_time = time.time() - start_time
        total_response_time += response_time
        total_requests += 1

        if response.status_code == 200:
            successful_requests += 1
        else:
            failed_requests += 1

        print(f"Status: {response.status_code}, Time taken: {response_time:.2f} seconds")
    
    except requests.RequestException as e:
        failed_requests += 1
        total_requests += 1
        print(f"Request failed: {e}")
    
    # Print statistics every 500 requests
    if total_requests % 500 == 0:
        avg_response_time = total_response_time / successful_requests if successful_requests > 0 else 0
        print(f"\n--- Request Statistics ---")
        print(f"Total Requests Sent: {total_requests}")
        print(f"Total Served Requests: {successful_requests}")
        print(f"Total Not Served Requests: {failed_requests}")
        print(f"Average Response Time: {avg_response_time:.2f} seconds\n")
        
def generate_poisson_requests(rate):
    while True:
        # Generate the interval based on Poisson distribution
        interval = np.random.poisson(rate)
        time.sleep(interval / 60)  # Convert interval to minutes
        send_request()

if __name__ == "__main__":
    generate_poisson_requests(60 / LAMBDA)
