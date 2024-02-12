"""
User Simulation Script

This script simulates user behavior by making HTTP requests to a specified endpoint at regular intervals.
It uses the requests library for HTTP communication and concurrent.futures for parallel execution of user simulations.

Constants:
- BASE_URL: The base URL of the endpoint to be accessed (http://[YOUR_IP_OF_APPLICATION_RUNNING_ON_WORKER]:8082/json_endpoint).

Functions:
- simulate_user(user_id, request_interval=1): Simulates a single user by making HTTP requests at a specified interval.
- run_simulation(num_users, interval, request_interval=1): Runs the overall simulation with multiple users and periodic restarts.

Usage:
- Set environment variables NUM_USERS, INTERVAL, and REQUEST_INTERVAL to configure the simulation parameters.
- NUM_USERS: Number of users (default is 10, capped at a maximum of 1000).
- INTERVAL: Time interval in minutes for periodic restarts of users (default is 1 minute).
- REQUEST_INTERVAL: Time interval in seconds between consecutive HTTP requests by each user (default is 1 second).
"""

import requests
import concurrent.futures
import time
import os

BASE_URL = 'http://localhost:8082/json_endpoint'

def simulate_user(user_id, request_interval=1):
    """
    Simulates a single user by making HTTP requests at a specified interval.

    Args:
        user_id (int): The identifier for the simulated user.
        request_interval (int): Time interval in seconds between consecutive HTTP requests.

    Returns:
        None
    """
    while True:
        try:
            response = requests.get(BASE_URL)
            print(f"User {user_id} - Status Code: {response.status_code}")
            time.sleep(request_interval)
        except Exception as e:
            print(f"User {user_id} - Error: {e}")

def run_simulation(num_users, interval, request_interval=1):
    """
    Runs the overall simulation with multiple users and periodic restarts.

    Args:
        num_users (int): Number of users to simulate (capped at a maximum of 1000).
        interval (int): Time interval in minutes for periodic restarts of users.
        request_interval (int): Time interval in seconds between consecutive HTTP requests.

    Returns:
        None
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        user_ids = range(1, num_users + 1)

        # Submit initial tasks for each user
        tasks = {executor.submit(simulate_user, user_id, request_interval): user_id for user_id in user_ids}

        while True:
            time.sleep(interval * 60)  # Convert minutes to seconds
            
            # Cancel existing tasks and submit new tasks to "restart" users
            for task in tasks:
                task.cancel()
            tasks = {executor.submit(simulate_user, user_id, request_interval): user_id for user_id in user_ids}

if __name__ == '__main__':
    num_users = int(os.environ.get('NUM_USERS', 10))
    interval = int(os.environ.get('INTERVAL', 1))
    request_interval = int(os.environ.get('REQUEST_INTERVAL', 1))

    if num_users > 1000:
        num_users = 1000

    run_simulation(num_users, interval, request_interval)
