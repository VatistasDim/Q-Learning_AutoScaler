import requests
import time

Applicationurl = 'http://application:8501/train'

while True:
    try:
        response = requests.get(Applicationurl)
        print(f"Response: {response}")
    except Exception as e:
        print(f"An error occurred: {e}, retry in 10 sec.")
        time.sleep(10)
