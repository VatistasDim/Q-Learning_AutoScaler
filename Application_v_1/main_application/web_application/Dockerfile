# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask app
EXPOSE 8082
EXPOSE 8000

# Command to start Flask app
CMD ["python", "app.py"]
