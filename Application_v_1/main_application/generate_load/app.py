from flask import Flask, render_template
from prometheus_client import start_http_server, Gauge
import time, psutil, threading

cpu_usage_gauge = Gauge('cpu_usage', 'CPU_Usage')
ram_usage_gauge = Gauge('ram_usage', 'Ram_Usage')
running_time_gauge = Gauge('running_time', 'Running Time')

app = Flask(__name__)

# def update_metrics():
#     cpu_usage_gauge.set(psutil.cpu_percent())
#     ram_usage_gauge.set(psutil.virtual_memory().percent)
#     running_time_gauge.set(time.time())
#     print("Data Output:\n CPU OUTPUT: "+cpu_usage_gauge+" RAM OUTPUT: "+ram_usage_gauge)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
    while True:
        cpu_usage_gauge.set(psutil.cpu_percent())
        ram_usage_gauge.set(psutil.virtual_memory().percent)
        running_time_gauge.set(time.time())
        print("Data Output:\n CPU OUTPUT: "+cpu_usage_gauge+" RAM OUTPUT: "+ram_usage_gauge)
        start_http_server(8000)
    # while True:
    #     metric_update_thread = threading.Thread(target=update_metrics)
    #     metric_update_thread.daemon = True
    #     metric_update_thread.start()
    #     time.sleep(1)