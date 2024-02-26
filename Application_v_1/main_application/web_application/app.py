"""
Flask Application with Prometheus Metrics and Endpoints.

This application includes routes for rendering HTML templates, serving video files,
and exposing a JSON endpoint. It also integrates Prometheus metrics for CPU usage,
RAM usage, and running time.

Metrics:
- cpu_usage: Gauge metric tracking CPU usage.
- ram_usage: Gauge metric tracking RAM usage.
- running_time: Gauge metric tracking the running time of the application.

Routes:
- /: Renders the home.html template.
- /about: Renders the about.html template.
- /graphics: Renders the graphics.html template.
- /streaming: Serves a video file (video_1.mp4) for streaming.
- /json_endpoint: Exposes a JSON endpoint, reading data from data.json.

"""

from flask import Flask, render_template, send_file, jsonify, request
from prometheus_client import start_http_server, Gauge, generate_latest
import time, psutil, threading, json


cpu_usage_gauge = Gauge('cpu_usage', 'CPU_Usage')
ram_usage_gauge = Gauge('ram_usage', 'Ram_Usage')
running_time_gauge = Gauge('running_time', 'Running Time')
response_time_gauge = Gauge('response_time', 'Response Time')
start_time = time.time()

app = Flask(__name__)

def update_metrics():
    """
    Continuously updates Prometheus metrics for CPU usage, RAM usage, and running time.

    This function runs in a separate thread and updates the metrics every second.

    Returns:
        None
    """
    while True:
        elapsed_time = time.time() - start_time
        cpu_usage_gauge.set(psutil.cpu_percent())
        ram_usage_gauge.set(psutil.virtual_memory().percent)
        running_time_gauge.set(int(elapsed_time))
        time.sleep(1)

@app.before_request
def before_request():
    """
    Called before a request is processed. Captures the start time of the request.

    Returns:
        None
    """
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """
    Called after a request is processed. Measures the response time and updates the metric.

    Args:
        response (flask.Response): The response object.

    Returns:
        flask.Response: The modified response object.
    """
    response_time = (time.time() - request.start_time)  # Elapsed time in seconds
    response_time_gauge.set(response_time)

    # Generate latest Prometheus metrics and reset response_time_gauge to 0
    generate_latest()
    response_time_gauge.set(0)

    return response

@app.route('/')
def home():
    """
    Renders the home.html template.

    Returns:
        str: Rendered HTML content.
    """
    return render_template('home.html')

@app.route('/about')
def about():
    """
    Renders the about.html template.

    Returns:
        str: Rendered HTML content.
    """
    return render_template('about.html')

@app.route('/graphics')
def graphics():
    """
    Renders the graphics.html template.

    Returns:
        str: Rendered HTML content.
    """
    return render_template('graphics.html')

@app.route('/streaming')
def streaming():
    """
    Serves a video file (video_1.mp4) for streaming.

    Returns:
        flask.Response: Response containing the video file.
    """
    video_path = 'static/video/video_1.mp4'
    return send_file(video_path, mimetype='video/mp4')

@app.route('/json_endpoint', methods=['GET'])
def json_endpoint():
    """
    Exposes a JSON endpoint, reading data from data.json.

    Returns:
        flask.Response: JSON response.
    """
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
