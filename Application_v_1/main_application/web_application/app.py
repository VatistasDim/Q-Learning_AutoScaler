from flask import Flask, render_template, send_file, jsonify, request
from prometheus_client import start_http_server, Gauge, Histogram, generate_latest
import time, psutil, threading, json

# Define Prometheus metrics
cpu_usage_gauge = Gauge('cpu_usage', 'CPU_Usage')
ram_usage_gauge = Gauge('ram_usage', 'Ram_Usage')
running_time_gauge = Gauge('running_time', 'Running Time')
cpu_shares_gauge = Gauge('cpu_shares', 'CPU Shares')
json_endpoint_histogram = Histogram('json_endpoint_response_time_seconds', 'Response time for JSON endpoint in seconds')

start_time = time.time()

app = Flask(__name__)

def get_cpu_shares():
    """
    Get the CPU shares assigned to the container.

    Returns:
        int: CPU shares value.
    """
    try:
        with open('/sys/fs/cgroup/cpu/cpu.shares', 'r') as file:
            cpu_shares = int(file.read().strip())
        return cpu_shares
    except FileNotFoundError:
        print("Error: CPU shares file not found.")
        return None
    except Exception as e:
        print("Error:", e)
        return None

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
        cpu_shares = get_cpu_shares()
        if cpu_shares is not None:
            cpu_shares_gauge.set(cpu_shares)
        time.sleep(1)

@app.route('/metrics')
def metrics():
    """
    Exposes Prometheus metrics.

    Returns:
        flask.Response: Prometheus metrics response.
    """
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

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
    response_time = time.time() - request.start_time  # Elapsed time in seconds
    
    # Check if the request path is for the JSON endpoint
    if request.path == '/json_endpoint':
        json_endpoint_histogram.observe(response_time)
        print(f"JSON endpoint response time: {response_time} seconds")

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
