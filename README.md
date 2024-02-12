Επεξήγηση του κώδικα.
Ξεκινόντας από τον φάκελο/project (main_application/load_balancer)

- Load Balancer:
  - Αρχείο: autoscaler_env.py:
 
      Το παρεχόμενο σενάριο αποτελεί ένα περιβάλλον OpenAI Gym για τον αυτόματο κλιμακοστασιμό ενός υπηρεσιών Docker, χρησιμοποιώντας μετρήσεις CPU και RAM για τον υπολογισμό ανταμοιβής. Ας εξετάσουμε τον κώδικα βήμα προς βήμα:
      
      ## Αρχική ρύθμιση και εισαγωγή βιβλιοθηκών:
      ```
      - python
      - import gym
      - import time
      - from gym import spaces
      - import numpy as np
      - import prometheus_metrics
      - import time, docker
      ```
      Εισάγονται οι απαραίτητες βιβλιοθήκες, συμπεριλαμβανομένων του OpenAI Gym, του Docker, του NumPy, των Prometheus metrics, και του χρόνου.
      
      ## Ρυθμίσεις υπηρεσίας Docker και cooldown:
      ```
      service_name = 'mystack_application'
      cooldownTimeInSec = 30
      
      client = docker.from_env()
      clients_list = client.services.list()
      Ορίζεται το όνομα της υπηρεσίας Docker και ο χρόνος αναμονής (cooldown).
      Δημιουργείται ένα αντικείμενο Docker client και ανακτάται η λίστα των υπηρεσιών Docker.
      Λειτουργίες Διαχείρισης Υπηρεσίας Docker:
      
      python
      def get_current_replica_count(service_prefix):
          # ...
      get_current_replica_count: Επιστρέφει τον αριθμό των αντιγράφων ενός Docker service με δοθέν πρόθεμα.
      python
      Copy code
      def scale_in_action(service_name, min_replicas):
          # ...
      def scale_out_action(service_name, max_replicas):
          # ...
      scale_in_action και scale_out_action: Υλοποιούν τη λογική για τη συρρίκνωση και επέκταση των αντιγράφων ενός Docker service αντίστοιχα.
      python
      Copy code
      def scale_out(service_name, desired_replicas):
          # ...
      def scale_in(service_name, scale_out_factor):
          # ...
      scale_out και scale_in: Συναρτήσεις για την κλιμάκωση ενός Docker service στον καθορισμένο αριθμό αντιγράφων.
      ```
      ## Υπολογισμός Ανταμοιβής και Λήψη Μετρητών:
      ```
      def get_reward(cpu_value, ram_value, cpu_threshold, ram_threshold):
          # ...
      get_reward: Υπολογίζει την ανταμοιβή με βάση τις μετρήσεις της CPU και της RAM και τους προκαθορισμένους κατωφλιακούς.
      python
      Copy code
      def fetch_data():
          # ...
      fetch_data: Ανακτά δεδομένα CPU, RAM και uptime από το API μετρήσεων του Prometheus.
      ```
      Επαναφορά Αντιγράφων και Υπολογισμός Κατωφλιών:
      
      ```
      def reset_replicas(service_name):
          # ...
      def Calculate_Thresholds():
          # ...
      reset_replicas: Επαναφέρει ένα Docker service σε ένα αντίγραφο.
      Calculate_Thresholds: Υπολογίζει τα κατώφλια CPU και RAM βάσει του τρέχοντος αριθμού αντιγράφων.
      ```
      Παραμετροποίηση Μετρητών Prometheus:
      ```
      url = 'http://prometheus:9090/api/v1/query'
      url: Η διεύθυνση URL για πρόσβαση στο API μετρήσεων του Prometheus.
      Ορισμός Περιβάλλοντος Gym:
      
      python
      Copy code
      class AutoscaleEnv(gym.Env):
          # ...
      ```
      ## AutoscaleEnv: Ορίζει ένα προσαρμοσμένο περιβάλλον OpenAI Gym για τον αυτόματο κλιμακοστασιμό μιας υπηρεσίας Docker.
      Τα χαρακτηριστικά περιλαμβάνουν το όνομα της υπηρεσίας, τα όρια των αντιγράφων, τα κατώφλια, και παραμέτρους περιβάλλοντος.
      
      Αρχικοποίηση Περιβάλλοντος και Δράσεις:
      ```
      def __init__(self, service_name, min_replicas, max_replicas, cpu_threshold, ram_threshold, num_states, max_time_minutes=10):
          # ...
      Αρχικοποιεί το περιβάλλον με τις καθορισμένες παραμέτρους, δράσεις και χώρους παρατήρησης.
      ```
      ## Δράσεις Περιβάλλοντος (Κλιμάκωση και Επαναφορά):
      
      ```
      def reset(self):
          # ...
      def step(self, action):
          # ...
      ```
      reset: Επαναφέρει το περιβάλλον στην αρχική του κατάσταση.
      step: Παίρνει μια δράση (κλιμάκωση ή σύρρικνωση), παρατηρεί τη νέα κατάσταση, υπολογίζει την ανταμοιβή και ελέγχει αν το επεισόδιο ολοκληρώθηκε.
      Παρατήρηση και Συνθήκες Λήξης:
      ```
      def _get_observation(self):
          # ...
      def _is_done(self):
          # ...
      ```
      _get_observation: Επιστρέφει τις τρέχουσες τιμές CPU και RAM ως παρατήρηση.
      _is_done: Ελέγχει αν το επεισόδιο ολοκληρώθηκε βάσει του χρόνου που έχει παρέλθει.
      Συνοπτικά, ο κώδικας παρέχει ένα ευέλικτο περιβάλλον OpenAI Gym που επιτρέπει την εκπαίδευση ενός μοντέλου ενισχυτικής μάθησης για τον αυτόματο κλιμακοστασιμό υπηρεσιών Docker, βάσει των μετρήσεων CPU και RAM.
        
        
