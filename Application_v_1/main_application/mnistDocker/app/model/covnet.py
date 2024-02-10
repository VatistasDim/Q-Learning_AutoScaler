import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
import psutil, time
from prometheus_client import start_http_server, Gauge


# Start Prometheus server
start_http_server(8082)

# Initialize Prometheus metrics
cpu_usage_gauge = Gauge('cpu_usage', 'CPU Usage')
ram_usage_gauge = Gauge('ram_usage', 'RAM Usage')
running_time_gauge = Gauge('running_time', 'Running Time')
start_time = time.time()
def metrics_callback():
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    cpu_usage_gauge.set(cpu_percent)
    ram_usage_gauge.set(ram_percent)
    running_time_gauge.set(time.time() - start_time)
    
class ConvNet(nn.Module):
    __slots__ = ['conv1', 'conv2', 'fc']

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=48,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=48,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def compile(self, criterion, optimizer) -> None:
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, trainloader, epochs) -> Tuple[int, float, float]:
        self.train()
        _padding = len(str(epochs))
        accuracy = 0.0
        for epoch in range(epochs):
            total = 0.0
            correct = 0.0
            with tqdm(trainloader, unit="batch", position=0, leave=True) as tepoch:
                for data, target in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1:<{_padding}}/{epochs}")
                    self.optimizer.zero_grad()
                    output = self.forward(data)
                    loss = self.criterion(output, target)
                    _, predicted = torch.max(output, 1)
                    loss.backward()
                    self.optimizer.step()
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    accuracy = correct / total
                    # Call the metrics callback if provided
                    metrics_callback()
                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                    loss = loss.item()
        return epochs, accuracy, loss

    def test(self, testloader):
        total = 0
        correct = 0
        self.eval()
        total_predicted = []
        with torch.no_grad():
            with tqdm(testloader, unit="batch", position=0, leave=True) as tepoch:
                for data, target in tepoch:
                    output = self.forward(data)
                    _, predicted = torch.max(output.data, 1)
                    total_predicted.append(predicted)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    accuracy=100. * (correct / total)
                    tepoch.set_postfix(accuracy=accuracy)

        print(f'\nFinished Testing with accuracy={100 * (correct / total):.2f}')
        return total_predicted, accuracy
