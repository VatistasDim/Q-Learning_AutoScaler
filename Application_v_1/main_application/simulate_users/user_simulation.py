import requests
import concurrent.futures
import time
import os

BASE_URL = 'http://localhost:8082/json_endpoint'

def simulate_user(user_id, request_interval=1):
    while True:
        try:
            response = requests.get(BASE_URL)
            print(f"User {user_id} - Status Code: {response.status_code}")
            time.sleep(request_interval)
        except Exception as e:
            print(f"User {user_id} - Error: {e}")

def run_simulation(num_users, interval, request_interval=1):
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
