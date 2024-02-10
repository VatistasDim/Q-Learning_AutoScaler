from flask import Flask, render_template, send_file, jsonify
from prometheus_client import start_http_server, Gauge
import time, psutil, threading, json

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
    video_path = 'static/video/video_1.mp4'
    return send_file(video_path, mimetype='video/mp4')

@app.route('/json_endpoint', methods=['GET'])
def json_endpoint():
    file_path = 'json_data/data.json'
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': 'JSON file not found'}), 404
    except json.JSONDecodeError:
        return jsonify({'error': 'Error decoding JSON file'}), 500
    
if __name__ == '__main__':
    
    metric_update_thread = threading.Thread(target=update_metrics)
    metric_update_thread.daemon = True
    metric_update_thread.start()
    
    start_http_server(8000)
    
    app.run(host='0.0.0.0', port=8082)