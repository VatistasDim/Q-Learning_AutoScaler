# Build Docker image

```bash
docker build -t myimage .
```

# Run Docker image

```bash
docker run -d --name mycontainer -p 8501:8501 myimage
```

localhost:8501/docs should show the swagger of the application
