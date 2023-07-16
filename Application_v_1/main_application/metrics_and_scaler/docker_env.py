import docker

def scale_service(service_name, desired_replicas):
    client = docker.from_env()
    print("DOCKER_ENV ==> ",client)
    service = client.services.get(service_name)
    print("DOCKER_SERVICE => ",service)
    service.update(mode=service.attrs['Spec']['Mode'], replicas=desired_replicas)

    print(f"Service '{service_name}' scaled to {desired_replicas} replicas.")

# if __name__ == '__main__':

#     service_name = 'my-service'


#     desired_replicas = 5

#     scale_service(service_name, desired_replicas)
