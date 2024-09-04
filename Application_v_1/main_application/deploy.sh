#!/bin/bash

clear
docker stack rm mystack
docker rmi load-balancer:latest --force
cd load_balancer/
docker build -t load-balancer ./
cd ..
docker stack deploy -c docker-compose.yml mystack
