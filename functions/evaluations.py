import carla
import cv2
import numpy as np
import math
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import queue

# --- Configuration ---
NEAR_CRASH_DISTANCE_3D = 2.0  # Ground Truth: 2.0 meters
NEAR_CRASH_DISTANCE_2D = 50   # Prediction: 50 pixels between box centers
YOLO_MODEL_PATH = 'yolov8n.pt'  # Your trained YOLOv8 model

# Lists to store our results
y_true = [] # Ground Truth labels
y_pred = [] # Prediction labels

def run_evaluation(y_true, y_pred):
    print("\n - Results -")

    if 1 not in y_true:
        print("No positive near-crash events")
        if 1 not in y_pred:
            print("Didnt predict anything")
            return
        else:
            print("false positive")

    labels = [0, 1]
    cm = confustion_matrix(y_true, y_pred, labels=labels)

    print("Confusion Matrix:")
    print(f"        [Predicted Safe] [Predicted Near Crash]")
    print(f"[Actual Safe]    {cm[0][0]:<16} {cm[0][1]:<20} (False Positive)")
    print(f"[Actual Near Crash] {cm[1][0]:<16} {cm[1][1]:<20} (True Positive)")

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                  pos_label = 1,
                                                  average='binary',
                                                  zero_division = 0)

    print("\n--- Key Metrics (for 'Near Crash' class) ---")
    print(f"🎯 Precision: {p:.2f}")
    print(f"   (Of all 'Near Crash' predictions, {p*100:.0f}% were correct.)")

    print(f"🎣 Recall:    {r:.2f}")
    print(f"   (Your model caught {r*100:.0f}% of all *real* 'Near Crash' events.)")

    print(f"⚖️ F1-Score:  {f1:.2f}")
    print(f"   (The balanced average of Precision and Recall.)")

def get_closest_pair(world):
    """
    Finds the single closest vehicle-pedestrian pair in the world.
    This is a simple way to focus on the most likely event.
    """
    vehicles = world.get_actors().filter('vehicle.*')
    walkers = world.get_actors().filter('walker.*')

    if not vehicles or not walkers:
        return None, None, float('inf') # Return infinity distance if no pair

    closest_dist = float('inf')
    closest_pair = (None, None)

    for v in vehicles:
        for w in walkers:
            dist = v.get_location().distance(w.get_location())
            if dist < closest_dist:
                closest_dist = dist
                closest_pair = (v, w)

    return closest_pair[0], closest_pair[1], closest_dist

def get_ground_truth_label(world):
    """
    Step 1: Define Ground Truth Label (1 or 0)
    Uses CARLA's 3D data to get the *perfect* answer.
    """
    vehicle, walker, distance_3d = get_closest_pair(world)

    if vehicle is None:
        return 0 # No event

    # Our GT Rule
    if distance_3d < NEAR_CRASH_DISTANCE_3D:
        return 1 # "Near Crash"
    else:
        return 0 # "Safe"

def get_prediction_label(image, yolo_model):
    """
    Step 2: Define Prediction Label (1 or 0)
    Uses YOLO's 2D data to make a *guess*.
    """
    # 1. Run YOLO detection
    results = yolo_model.predict(image, verbose=False)

    # 2. Find boxes for 'person' (class 0) and 'car' (class 2)
    # Note: Class IDs may vary based on your model. 0=person, 2=car in standard COCO.
    person_boxes = []
    car_boxes = []

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0: # 0 is 'person'
                person_boxes.append(box.xyxy[0])
            elif int(box.cls) in [2, 5, 7]: # 2=car, 5=bus, 7=truck
                car_boxes.append(box.xyxy[0])

    if not person_boxes or not car_boxes:
        return 0 # Can't see both, so no "near crash"

    # 3. Find the closest 2D pair
    # For simplicity, we'll just use the first one found
    p_box = person_boxes[0]
    c_box = car_boxes[0]

    # Calculate centers of the 2D boxes
    p_center = ((p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2)
    c_center = ((c_box[0] + c_box[2]) / 2, (c_box[1] + c_box[3]) / 2)

    # Calculate 2D Euclidean distance
    distance_2d = math.sqrt((p_center[0] - c_center[0])**2 + (p_center[1] - c_center[1])**2)

    # 4. Our Prediction Rule
    if distance_2d < NEAR_CRASH_DISTANCE_2D:
        return 1 # "Near Crash"
    else:
        return 0 # "Safe"

def process_image(image):
    """Callback function for the camera sensor."""
    # Convert CARLA's raw BGRA image to an RGB numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3] # Drop alpha channel
    image_queue.put(array) # Add to queue for processing

# --- Main Execution ---

# 1. Setup CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()
original_settings = world.get_settings()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1 # 10 FPS. Good for balancing sim and ML.
world.apply_settings(settings)

# 2. Setup YOLO Model
model = YOLO(YOLO_MODEL_PATH)

# 3. Setup CARLA Sensor (RGB Camera)
blueprint_library = world.get_blueprint_library()
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
# Spawn camera attached to a vehicle (or set a fixed spectator view)
# ... (This assumes you have an 'ego_vehicle' spawned)
# transform = carla.Transform(carla.Location(x=1.5, z=2.4))
# camera = world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle)

# --- FOR THIS EXAMPLE: We'll attach the camera to the spectator ---
spectator = world.get_spectator()
transform = spectator.get_transform()
camera = world.spawn_actor(camera_bp, transform, attach_to=spectator)

# 4. Setup Image Queue
image_queue = queue.Queue()
camera.listen(process_image)

actor_list = [camera] # Keep track of spawned actors to destroy later

try:
    print("Running simulation... Press Ctrl+C to stop.")
    # 5. Run the Main Loop
    for frame in range(500): # Run for 500 frames
        # --- A. Advance the simulation ---
        world.tick()

        # --- B. Get the camera image ---
        try:
            image = image_queue.get(timeout=1.0)
        except queue.Empty:
            print(f"Frame {frame}: Missed image from queue")
            continue

        # --- C. Get Ground Truth Label (The "Answer") ---
        gt_label = get_ground_truth_label(world)

        # --- D. Get Prediction Label (The "Guess") ---
        pred_label = get_prediction_label(image, model)

        # --- E. Store the results ---
        y_true.append(gt_label)
        y_pred.append(pred_label)

        print(f"Frame {frame}: Ground Truth={gt_label}, Prediction={pred_label}")

        # (Optional) Display the image with OpenCV
        # cv2.imshow('CARLA Feed', image)
        # if cv2.waitKey(1) == ord('q'):
        #     break

finally:
    # 6. Cleanup
    print("\nCleaning up simulation...")
    world.apply_settings(original_settings)
    for actor in actor_list:
        actor.destroy()
    # cv2.destroyAllWindows()

    print("Simulation finished. Running evaluation...")

    # --- This is the start of Phase 2 ---
    run_evaluation(y_true, y_pred)
