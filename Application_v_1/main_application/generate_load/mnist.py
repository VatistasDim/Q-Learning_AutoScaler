import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from prometheus_client import start_http_server, Counter, Gauge
import time, psutil

counter = Counter('mnist_counter', 'MNIST Counter')
cpu_usage_gauge = Gauge('mnist_cpu_usage', 'CPU_Usage')
ram_usage_gauge = Gauge('mnist_ram_usage', 'Ram_Usage')
running_time_gauge = Gauge('mnist_running_time', 'Running Time')

start_http_server(8000)

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start_time = time.time()
while True:

    model.fit(x_train, y_train, epochs=10, batch_size=64)

    counter.inc()
    
    cpu_usage_gauge.set(psutil.cpu_percent())
    
    ram_usage_gauge.set(psutil.virtual_memory().percent)
    
    running_time_gauge.set(time.time() - start_time)
    
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
    time.sleep(1)