# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install required packages
RUN pip install Flask prometheus_client psutil

# Expose port for Flask app
EXPOSE 8082

# Command to start Flask app
CMD ["python", "-u", "app.py"]
