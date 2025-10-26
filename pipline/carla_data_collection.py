import os
import json
import random
import shutil
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

class CARLADataCollector:
    """Collects synthetic driving data from CARLA simulator"""

    def __init__(self, output_dir='carla_dataset', close_call_threshold=5.0):
        self.output_dir = output_dir
        self.close_call_threshold = close_call_threshold
        self.setup_directories()

    def setup_directories(self):
        """Create directory structure"""
        dirs = ['raw_images', 'annotations', 'metadata']
        for d in dirs:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

    def connect_to_carla(self, host='localhost', port=2000):
        """Connect to CARLA simulator"""
        try:
            import carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            print(f"✓ Connected to CARLA simulator at {host}:{port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to CARLA: {e}")
            print("Make sure CARLA simulator is running!")
            return False

    def spawn_ego_vehicle_with_camera(self):
        """Spawn the main vehicle with attached camera"""
        import carla

        # Spawn ego vehicle
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.ego_vehicle.set_autopilot(True)

        # Attach camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')

        camera_transform = carla.Transform(carla.Location(x=2.5, z=1.5))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform,
                                            attach_to=self.ego_vehicle)

        self.image_data = {'frame': None}
        self.camera.listen(lambda image: self._process_image(image))

        print("✓ Ego vehicle and camera spawned")

    def _process_image(self, image):
        """Process incoming camera images"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.image_data['frame'] = array[:, :, :3]

    def spawn_traffic(self, num_vehicles=50):
        """Spawn traffic vehicles"""
        import carla

        self.traffic_vehicles = []
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for i in range(min(num_vehicles, len(spawn_points))):
            vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[i])

            if vehicle:
                vehicle.set_autopilot(True)
                self.traffic_vehicles.append(vehicle)

        print(f"✓ Spawned {len(self.traffic_vehicles)} traffic vehicles")

    def calculate_distances_and_bboxes(self):
        """Calculate distances to nearby vehicles and their 2D bounding boxes"""
        import carla

        ego_location = self.ego_vehicle.get_location()
        ego_transform = self.ego_vehicle.get_transform()

        camera_transform = self.camera.get_transform()
        camera_matrix = self._build_projection_matrix()

        vehicles_data = []

        for vehicle in self.traffic_vehicles:
            if vehicle.id == self.ego_vehicle.id:
                continue

            vehicle_location = vehicle.get_location()
            distance = ego_location.distance(vehicle_location)

            # Only process nearby vehicles
            if distance < 50.0:
                # Get 3D bounding box
                bbox = vehicle.bounding_box
                vertices = bbox.get_world_vertices(vehicle.get_transform())

                # Project to 2D
                bbox_2d = self._get_2d_bbox(vertices, camera_matrix, camera_transform)

                if bbox_2d:
                    vehicles_data.append({
                        'distance': distance,
                        'bbox_2d': bbox_2d,
                        'is_close_call': distance < self.close_call_threshold,
                        'vehicle_type': vehicle.type_id
                    })

        return vehicles_data

    def _build_projection_matrix(self, fov=90):
        """Build camera projection matrix"""
        width = 1280
        height = 720
        focal = width / (2.0 * np.tan(fov * np.pi / 360.0))

        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0

        return K

    def _get_2d_bbox(self, vertices, K, camera_transform):
        """Project 3D bounding box to 2D image coordinates"""
        import carla

        points_2d = []

        for vertex in vertices:
            # World to camera coordinates
            point = np.array([vertex.x, vertex.y, vertex.z, 1.0])

            # Transform to camera space
            camera_matrix = np.array(camera_transform.get_matrix())
            camera_inv = np.linalg.inv(camera_matrix)
            point_camera = camera_inv.dot(point)

            # Skip points behind camera
            if point_camera[1] < 0:
                return None

            # Project to 2D
            point_2d = K.dot(point_camera[:3])
            point_2d = point_2d[:2] / point_2d[2]
            points_2d.append(point_2d)

        if not points_2d:
            return None

        points_2d = np.array(points_2d)
        x_min, y_min = points_2d.min(axis=0)
        x_max, y_max = points_2d.max(axis=0)

        # Clip to image boundaries
        x_min = max(0, min(x_min, 1280))
        x_max = max(0, min(x_max, 1280))
        y_min = max(0, min(y_min, 720))
        y_max = max(0, min(y_max, 720))

        # Check if bbox is valid
        if x_max <= x_min or y_max <= y_min:
            return None

        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def collect_data(self, num_frames=1000, save_interval=10):
        """Main data collection loop"""
        import carla
        import time

        frame_count = 0
        saved_count = 0

        print(f"\n🎬 Starting data collection ({num_frames} frames)...")
        print(f"Close call threshold: {self.close_call_threshold}m\n")

        try:
            while frame_count < num_frames:
                self.world.tick()

                if self.image_data['frame'] is None:
                    continue

                if frame_count % save_interval == 0:
                    # Get current frame
                    image = self.image_data['frame'].copy()

                    # Calculate vehicle data
                    vehicles_data = self.calculate_distances_and_bboxes()

                    # Determine if this is a close call scenario
                    is_close_call = any(v['is_close_call'] for v in vehicles_data)

                    # Save data
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    prefix = "close_call" if is_close_call else "safe"
                    filename = f"{prefix}_{timestamp}_{saved_count:06d}"

                    # Save image
                    img_path = os.path.join(self.output_dir, 'raw_images', f"{filename}.jpg")
                    cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                    # Save metadata
                    metadata = {
                        'filename': filename,
                        'is_close_call': is_close_call,
                        'vehicles': vehicles_data,
                        'frame_number': frame_count
                    }

                    metadata_path = os.path.join(self.output_dir, 'metadata', f"{filename}.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    saved_count += 1

                    if saved_count % 50 == 0:
                        close_call_count = sum(1 for v in vehicles_data if v['is_close_call'])
                        print(f"Saved {saved_count}/{num_frames//save_interval} | "
                              f"Frame {frame_count} | "
                              f"Close calls in frame: {close_call_count}")

                frame_count += 1

        except KeyboardInterrupt:
            print("\n\n⚠ Data collection interrupted by user")

        print(f"\n✓ Data collection complete! Saved {saved_count} images")
        return saved_count

    def cleanup(self):
        """Clean up CARLA actors"""
        try:
            if hasattr(self, 'camera'):
                self.camera.destroy()
            if hasattr(self, 'ego_vehicle'):
                self.ego_vehicle.destroy()
            if hasattr(self, 'traffic_vehicles'):
                for vehicle in self.traffic_vehicles:
                    vehicle.destroy()
            print("✓ Cleaned up CARLA actors")
        except Exception as e:
            print(f"Warning during cleanup: {e}")


COLLECT_DATA = False  # Set to True if you have CARLA running

if COLLECT_DATA:
    print("\n" + "=" * 70)
    print("STEP 1: DATA COLLECTION FROM CARLA")
    print("=" * 70)

    collector = CARLADataCollector(
        output_dir='carla_dataset',
        close_call_threshold=5.0
    )

    if collector.connect_to_carla():
        collector.spawn_ego_vehicle_with_camera()
        collector.spawn_traffic(num_vehicles=50)
        collector.collect_data(num_frames=1000, save_interval=10)
        collector.cleanup()
    else:
        print("\n⚠ Skipping data collection - CARLA not available")
        print("You can use pre-collected data or run CARLA simulator")
