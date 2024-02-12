Επεξήγηση του κώδικα.
Ξεκινόντας από τον φάκελο/project (main_application/load_balancer)

- Load Balancer (Φάκελος/Project):
## Αρχείο: autoscaler_env.py:
 
Αυτό το σενάριο Python παρέχει ένα περιβάλλον για την αυτόματη κλιμάκωση ενός Docker service χρησιμοποιώντας το OpenAI Gym. Το περιβάλλον περιλαμβάνει ενέργειες για κλιμάκωση προς τα πάνω και προς τα κάτω, ενώ υπολογίζει επιβράβευση βασισμένη σε μετρήσεις CPU και RAM.
## Χρήση:

- Δημιουργήστε μια νέα έκδοση της κλάσης AutoscaleEnv παρέχοντας τις απαραίτητες παραμέτρους.
- Χρησιμοποιήστε τη μέθοδο reset() για να αρχικοποιήσετε το περιβάλλον.
- Χρησιμοποιήστε τη μέθοδο step(action) για να πάρετε μια ενέργεια (0 για κλιμάκωση προς τα πάνω, 1 για κλιμάκωση προς τα κάτω) και να παρατηρήσετε τη νέα κατάσταση, την ανταμοιβή, και αν το περιβάλλον είναι ολοκληρωμένο.
## Σημαντικά Σημεία:

- Η κλάση AutoscaleEnv κληρονομεί από την κλάση gym.Env του OpenAI Gym.
- Το περιβάλλον προσομοιώνει μια συνεχή κατάσταση με δύο δυνατές ενέργειες: κλιμάκωση προς τα πάνω (scale_out) και κλιμάκωση προς τα κάτω (scale_in).
- Οι μετρήσεις για το CPU και το RAM ανακτώνται από ένα σύστημα παρακολούθησης Prometheus.
- Η ανταμοιβή υπολογίζεται βάσει των μετρήσεων CPU και RAM σε σχέση με καθορισμένα όρια.
- Ο κώδικας περιλαμβάνει λειτουργίες όπως scale_in_action, scale_out_action, fetch_data, get_reward, reset_replicas, κ.ά.
```
import gym
import time
from gym import spaces
import numpy as np
import prometheus_metrics
import time, docker

# ... (Ολόκληρος ο κώδικας περιβάλλοντος)

# Παράδειγμα χρήσης:
env = AutoscaleEnv(
    service_name='mystack_application',
    min_replicas=1,
    max_replicas=10,
    cpu_threshold=70,
    ram_threshold=80,
    num_states=2,
    max_time_minutes=10
)

observation = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(_ + 1))
        break
```
## Υπολογισμός Κατωφλίων:

Το threshold για CPU και RAM υπολογίζεται με βάση τον αριθμό των επαναλήψεων (αντιγράφων) της υπηρεσίας Docker. Αναφέρεται ως cpu_threshold και ram_threshold στον κώδικα.

Οι λειτουργίες Calculate_Thresholds() και get_reward() είναι υπεύθυνες για τον υπολογισμό αυτών των τιμών.
Εδώ είναι πώς υπολογίζονται:

1. Calculate_Thresholds(): Αυτή η συνάρτηση υπολογίζει τα όρια cpu_threshold και ram_threshold βάσει του τρέχοντος αριθμού αντιγράφων της υπηρεσίας Docker.
   ```
   def Calculate_Thresholds():
    current_replicas = get_current_replica_count(service_name)
    if current_replicas is not None:
        cpu_threshold = 1 + (current_replicas - 1) * 8 if current_replicas <= 10 else 100
        ram_threshold = 10 + (current_replicas - 1) * 8 if current_replicas <= 10 else 100
    else:
        cpu_threshold = 0  # Default value if replicas count is not available
        ram_threshold = 10  # Default value if replicas count is not available

    print(f"Thresholds calculated as CPU:{cpu_threshold}, RAM: {ram_threshold}")
    return cpu_threshold, ram_threshold
   ```
2. get_reward(): Αυτή η συνάρτηση υπολογίζει την ανταμοιβή βάσει των τρεχουσών τιμών CPU και RAM σε σχέση με τα καθορισμένα όρια.
   ```
   def get_reward(cpu_value, ram_value, cpu_threshold, ram_threshold):
    if cpu_value is not None and ram_value is not None:
        are_too_many_containers = False
        close_to_achieve_reward = False
        cpu_threshold_20_percent = cpu_value * 0.20
        ram_threshold_20_percent = ram_value * 0.20
        cpu_threshold_merged = cpu_threshold + int(cpu_threshold_20_percent)
        ram_threshold_merged = ram_threshold + int(ram_threshold_20_percent)

        cpu_diff = cpu_threshold_merged - cpu_value
        cpu_diff_low = cpu_value - cpu_threshold_merged
        if cpu_diff >= 10:
            are_too_many_containers = True
        elif cpu_diff_low <= 15:
            close_to_achieve_reward = True
        
        if cpu_value <= cpu_threshold_merged and ram_value <= ram_threshold_merged:
            print(f"Reward={20}, cpu_value={cpu_value} <= {cpu_threshold_merged} and ram_value={ram_value} <= {ram_threshold_merged}")
            return 20
        elif close_to_achieve_reward:
            if are_too_many_containers:
                print(f"Reward {-10}: Caused by, Too many containers are running.")
                return -10
            print("Close to achieve reward!")
            print(f"Reward{-5}, cpu_value={cpu_value} <= {cpu_threshold_merged} and ram_value={ram_value} <= {ram_threshold_merged}")
            return -5
        elif are_too_many_containers:
            print(f"Reward {-10}: Caused by, Too many containers are running.")
            return -10
        else:
            print(f"Reward={-15}, cpu_value={cpu_value} >= {cpu_threshold_merged} or ram_value={ram_value} >= {ram_threshold_merged}")
            return -15
    else:
        print(f"Reward={0}: Caused by, There was an error when trying to calculate the reward function")
        return 0
   ```
Τα παραπάνω κομμάτια κώδικα χρησιμοποιούν διάφορες παραμέτρους και υπολογισμούς, όπως το ποσοστό 20% (cpu_threshold_20_percent, ram_threshold_20_percent) για να δημιουργήσουν ένα "περιθώριο" κατώτερου και ανώτερου όριου, καθώς και να ελέγξουν την απόκλιση από αυτά τα όρια. Οι τιμές των ανταμοιβών εκτυπώνονται επίσης για να βοηθήσουν στην κατανόηση του λόγου χορήγησής τους.

Σημείωση: Ο κώδικας είναι μακρύς και περιλαμβάνει πολλές λειτουργίες. Η παραπάνω επισκόπηση αφαιρεί τα μέρη του κώδικα, για να εστιάσει στη χρήση και τη δομή του περιβάλλοντος.

## Αρχείο: main.py:

Ο κώδικας περιλαμβάνει έναν ελέγχου κύκλο που παρακολουθεί τις μετρήσεις CPU και RAM, λαμβάνει συγκεκριμένες ενέργειες βάσει κατωφλίων και συνθηκών, ενημερώνει ένα Q-τιμή, και διαχειρίζεται λειτουργίες κλιμάκωσης με βάση την τρέχουσα κατάσταση και τις ενέργειες.
Παρακάτω εξηγούνται ορισμένες σημαντικές λειτουργίες και μέρη του κώδικα:
discretize_state(cpu_value, ram_value): Αυτή η συνάρτηση χρησιμοποιείται για να διαιρέσει τις τιμές της CPU και της RAM σε διακριτές καταστάσεις.

1. select_action(Q, cpu_state, ram_state): Αυτή η συνάρτηση επιλέγει μια ενέργεια (scale_in ή scale_out) βάσει των Q-τιμών και των τρεχουσών καταστάσεων CPU και RAM.

2. update_q_value(Q, state, action, reward, next_state): Αυτή η συνάρτηση ενημερώνει τις Q-τιμές βάσει της τρέχουσας κατάστασης, της ενέργειας, της ανταμοιβής και της επόμενης κατάστασης.

3. fetch_data(): Αυτή η συνάρτηση ανακτά τα δεδομένα των μετρήσεων CPU, RAM και χρόνου εκτέλεσης από το Prometheus.

4. calculate_mse(Q, target_values): Αυτή η συνάρτηση υπολογίζει το Mean Squared Error (MSE) μεταξύ των Q-τιμών και των τιμών προορισμού.

5. plot_values(iterations, mse_values, save_path): Αυτή η συνάρτηση δημιουργεί γραφήματα MSE κατά τη διάρκεια των επαναλήψεων.

Επιπρόσθετα, ο κώδικας περιλαμβάνει έναν κύριο έλεγχο (if __name__ == "__main__":) όπου γίνεται η εκτέλεση του κώδικα. Κατά τη διάρκεια κάθε επανάληψης, ελέγχονται οι μετρήσεις CPU και RAM, λαμβάνεται μια ενέργεια βάσει του Q-learning, ενημερώνονται οι Q-τιμές, και υπολογίζεται το MSE. Επίσης, υπάρχει μια φάση επικύρωσης για την εκτίμηση του αλγόριθμου κατά τη διάρκεια της εκπαίδευσης.

Ο κώδικας χρησιμοποιεί την Q-Learning για την εκμάθηση βέλτιστης στρατηγικής (πολιτικής) για την αυτόματη κλιμάκωση (autoscaling) ενός Docker service. Εδώ θα εξηγήσω τα βασικά στοιχεία της Q-Learning στον παραδοθέντα κώδικα:

1. Πίνακας Q:
   ```
   Q = np.zeros((num_states, num_states, 2))
   ```
Ο πίνακας Q αναπαριστά τις τιμές της πολιτικής (ποια ενέργεια να επιλέξει) για κάθε δυνατό ζεύγος (κατάσταση, ενέργεια). Στην περίπτωση αυτή, έχουμε 2 ενέργειες (scale_out και scale_in) και 2 διακριτές καταστάσεις (CPU state και RAM state).

2. Επιλογή Ενέργειας:
  ```
  action = select_action(Q, cpu_state, ram_state)
  ```
Η συνάρτηση select_action επιλέγει μια ενέργεια βάσει του πίνακα Q. Υπάρχει ένας βαθμός εξερεύνησης (epsilon), που καθορίζει πόσο συχνά ο αλγόριθμος θα πραγματοποιεί τυχαίες ενέργειες αντί να επιλέγει τη βέλτιστη ενέργεια.

3. Ενημέρωση Q-Τιμών:
  ```
  update_q_value(Q, (cpu_state, ram_state), action, reward, (next_cpu_state, next_ram_state))
  ```
Η συνάρτηση update_q_value υπολογίζει το νέο Q-τιμή βάσει του κανόνα ενημέρωσης Q. Οι παράμετροι είναι η προηγούμενη κατάσταση, η προηγούμενη ενέργεια, η ανταμοιβή που λήφθηκε και η νέα κατάσταση.

Η Q-Learning λειτουργεί με την ιδέα ότι ο πίνακας Q συγκεντρώνει τις βέλτιστες πολιτικές που έχουν εκτιμηθεί από τον αλγόριθμο καθώς προχωρά στον χώρο καταστάσεων και ενεργειών. Καθώς επαναλαμβάνονται οι επαναλήψεις και λαμβάνονται ανταμοιβές από το περιβάλλον, ο πίνακας Q ενημερώνεται για να αντικατοπτρίζει τις βέλτιστες ενέργειες σε κάθε κατάσταση.

- Simulate_users (Φάκελος/Project):
  ## Αρχείο: simulate_users.py:
Αυτό το σενάριο προσομοιώνει τη συμπεριφορά του χρήστη με τη δημιουργία HTTP αιτημάτων σε ένα καθορισμένο σημείο πρόσβασης (endpoint) σε τακτικά χρονικά διαστήματα. Χρησιμοποιεί τη βιβλιοθήκη requests για την επικοινωνία HTTP και το concurrent.futures για την παράλληλη εκτέλεση των προσομοιώσεων χρηστών.
### Σταθερές:

- BASE_URL: Η βασική διεύθυνση URL του σημείου πρόσβασης που πρόκειται να αποκαλυφθεί.
### Συναρτήσεις:

- simulate_user(user_id, request_interval=1): Προσομοιώνει τη συμπεριφορά ενός μόνο χρήστη κάνοντας HTTP αιτήματα σε καθορισμένο χρονικό διάστημα.
run_simulation(num_users, interval, request_interval=1): Εκτελεί τη συνολική προσομοίωση με πολλούς χρήστες και περιοδικές επανεκκινήσεις.
### Χρήση:

- Ορίστε τις μεταβλητές περιβάλλοντος NUM_USERS, INTERVAL και REQUEST_INTERVAL για να ρυθμίσετε τις παραμέτρους προσομοίωσης.
- NUM_USERS: Αριθμός χρηστών (προεπιλεγμένος είναι 10, με μέγιστο όριο 1000).
- INTERVAL: Χρονικό διάστημα σε λεπτά για περιοδικές επανεκκινήσεις χρηστών (προεπιλεγμένο είναι 1 λεπτό).
- REQUEST_INTERVAL: Χρονικό διάστημα σε δευτερόλεπτα μεταξύ διαδοχικών HTTP αιτημάτων από κάθε χρήστη (προεπιλεγμένο είναι 1 δευτερόλεπτο).

- web_application (Φάκελος/Project):
  ## Αρχείο: app.py:

Αυτή η εφαρμογή Flask περιλαμβάνει διαδρομές για την απεικόνιση HTML προτύπων, την παροχή αρχείων βίντεο και την εκθεση ενός JSON σημείου πρόσβασης. Ενσωματώνει επίσης τα μετρήσιμα Prometheus για τη χρήση της CPU, τη χρήση της RAM και τον χρόνο εκτέλεσης.

## Μετρήσεις:

- cpu_usage: Μετρική κλίμακας παρακολούθησης της χρήσης της CPU.
- ram_usage: Μετρική κλίμακας παρακολούθησης της χρήσης της RAM.
- running_time: Μετρική κλίμακας παρακολούθησης του χρόνου εκτέλεσης της εφαρμογής.
## Διαδρομές:

- /: Απεικονίζει το πρότυπο home.html.
- /about: Απεικονίζει το πρότυπο about.html.
- /graphics: Απεικονίζει το πρότυπο graphics.html.
- /streaming: Παρέχει ένα αρχείο βίντεο (video_1.mp4) για ροή.
- /json_endpoint: Εκθέτει ένα JSON σημείο πρόσβασης, διαβάζοντας δεδομένα από το data.json.
## Κώδικας:
```
from flask import Flask, render_template, send_file, jsonify
from prometheus_client import start_http_server, Gauge
import time, psutil, threading, json

# Δημιουργία μετρητών Prometheus
cpu_usage_gauge = Gauge('cpu_usage', 'CPU_Usage')
ram_usage_gauge = Gauge('ram_usage', 'Ram_Usage')
running_time_gauge = Gauge('running_time', 'Running Time')

# Αρχικοποίηση χρονομέτρησης
start_time = time.time()

app = Flask(__name__)

# Συνάρτηση για συνεχή ενημέρωση των μετρήσεων Prometheus
def update_metrics():
    while True:
        elapsed_time = time.time() - start_time
        cpu_usage_gauge.set(psutil.cpu_percent())
        ram_usage_gauge.set(psutil.virtual_memory().percent)
        running_time_gauge.set(int(elapsed_time))
        time.sleep(1)

# Διαδρομή για απεικόνιση του προτύπου home.html
@app.route('/')
def home():
    return render_template('home.html')

# Διαδρομή για απεικόνιση του προτύπου about.html
@app.route('/about')
def about():
    return render_template('about.html')

# Διαδρομή για απεικόνιση του προτύπου graphics.html
@app.route('/graphics')
def graphics():
    return render_template('graphics.html')

# Διαδρομή για ροή ενός αρχείου βίντεο (video_1.mp4)
@app.route('/streaming')
def streaming():
    video_path = 'static/video/video_1.mp4'
    return send_file(video_path, mimetype='video/mp4')

# Διαδρομή για εκθεση JSON
@app.route('/json_endpoint', methods=['GET'])
def json_endpoint():
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
    # Ξεκίνημα νημάτων για τη συνεχή ενημέρωση των μετρήσεων Prometheus και του HTTP server
    metric_update_thread = threading.Thread(target=update_metrics)
    metric_update_thread.daemon = True
    metric_update_thread.start()

    start_http_server(8000)
    
    # Ξεκίνημα του Flask
    app.run(host='0.0.0.0', port=8082)
```
## Χρήση:

- Ρυθμίστε τις μεταβλητές περιβάλλοντος για να προσαρμόσετε τις παραμέτρους της εφαρμογής (π.χ., NUM_USERS, INTERVAL, REQUEST_INTERVAL).

  ## Αρχείο: docker-compose.yml:
Tο αρχείο docker-compose.yml περιγράφει τη διαμόρφωση για ένα σύνολο υπηρεσιών που χρησιμοποιούνται σε ένα σύστημα με Docker. Κάθε υπηρεσία αναφέρεται σε ένα περιβάλλον Docker και καθορίζει τις ρυθμίσεις για την εικόνα, τις πόρτες, τα όγκα, και άλλες παραμέτρους.

## Υπηρεσίες:

application (Εφαρμογή):

- Εικόνα: application
- Πόρτες: 8082 για την εφαρμογή και 8000 για το HTTP server της.
prometheus (Prometheus Μετρήσεις):

- Εικόνα: prom/prometheus:v2.30.3
-Πόρτα: 9090 για τον Prometheus server.
- Όγκοι: Κοινόχρηστος όγκος μεταξύ του host και του container για το prometheus.yml.
load-balancer (Φορτωτής Φορτίου):

- Εικόνα: load-balancer
- Όγκοι: Κοινόχρηστοι όγκοι μεταξύ του host και του container για αποθήκευση γραφικών και βαρών Q-learning.
- Κανόνες τοποθέτησης: Ορίζεται στον διαχειριστή (manager) κόμβο.
grafana (Grafana Dashboard):

- Εικόνα: grafana/grafana:latest
- Πόρτα: 3000 για τον Grafana server.
- Εξαρτήσεις: Εξαρτάται από το prometheus

```
  version: '3.1'
    
services:
  application:
    image: application
    ports:
      - "8082:8082"
      - "8000:8000"

  prometheus:
    image: prom/prometheus:v2.30.3
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
  
  load-balancer:
    image: load-balancer
    volumes:
      - ~/tutorial/Application_v_1/plots:/plots
      - ~/tutorial/Application_v_1/QSavedWeights:/QSavedWeights
      - /var/run/docker.sock:/var/run/docker.sock
    deploy:
      placement:
        constraints:
          - node.role == manager

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
  ```
## Χρήση:

Εκτελέστε το αρχείο docker-compose.yml με την εντολή docker-compose up.
Τοποθετήστε τα σχετικά αρχεία όπως το prometheus.yml στον κατάλογο prometheus στον κεντρικό φάκελο του project.
Οι υπηρεσίες θα εκκινήσουν και θα είναι διαθέσιμες στις καθορισμένες πόρτες.
