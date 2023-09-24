from flask import Flask, render_template
from prometheus_client import start_http_server, Gauge
import time, psutil, threading

cpu_usage_gauge = Gauge('cpu_usage', 'CPU_Usage')
ram_usage_gauge = Gauge('ram_usage', 'Ram_Usage')
running_time_gauge = Gauge('running_time', 'Running Time')
start_time = time.time()

app = Flask(__name__)

def update_metrics():
    while True:
        elapsed_time = time.time() - start_time
        cpu_usage_gauge.set(psutil.cpu_percent())
        ram_usage_gauge.set(psutil.virtual_memory().percent)
        running_time_gauge.set(int(elapsed_time))
        print("Data Output:\n CPU OUTPUT: "+ str(cpu_usage_gauge)+" RAM OUTPUT: "+str(ram_usage_gauge))
        time.sleep(1)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/graphics')
def graphics():
    return render_template('graphics.html')

@app.route('/streaming')
def streaming():
    return render_template('streaming.html')

if __name__ == '__main__':
    
    metric_update_thread = threading.Thread(target=update_metrics)
    metric_update_thread.daemon = True
    metric_update_thread.start()
    
    start_http_server(8000)
    
    app.run(host='0.0.0.0', port=8082)