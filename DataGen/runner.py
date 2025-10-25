import subprocess
import time
import carla
import os
import sys
import keyboard
import numpy as np
from PIL import Image
from queue import Queue, Empty

# For imports
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"📂 CWD: {os.getcwd()}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build a reliable relative path to your CARLA executable (optional launcher)
CARLA_EXE = os.path.join(BASE_DIR, "CARLA_0.9.16", "CarlaUE4.exe")

# Weather (you can tweak)
weather_settings = {
    "cloudiness": 0.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 0.0,
    "fog_density": 4.0,
    "fog_distance": 0.1,
    "wetness": 0.0,
    "fog_falloff": 0.0,
    "scattering_intensity": 0.03,
    "mie_scattering_scale": 0.0331,
    "rayleigh_scattering_scale": 0.0,
    "sun_altitude_angle": 45.0,
    "sun_azimuth_angle": 120.0,
}

# Traffic generation settings (counts are used; others handled in code)
traffic_settings = {
    "number-of-vehicles": 60,
    "number-of-walkers": 30,
    "car-lights-on": True,
    "hybrid": True,
}

# A safe default camera; your earlier huge coords were off-map
camera_config = [
    {'x': -30.093, 'y': 37.74, 'z': 14.396, 'pitch': -26.022, 'yaw': -136.872, 'roll': 0.0},
    {'x': -63.103, 'y': 2.151, 'z': 10.915, 'pitch': -28.45, 'yaw': 49.403, 'roll': 0},
    {'x': 80.42337, 'y': -15.39008, 'z': 9.2415, 'pitch': -6.9456, 'yaw': 58.167, 'roll': 0},  # Needs work
    {'x': 93.338, 'y': 59.921, 'z': 11.431, 'pitch': -5.587, 'yaw': -59.91, 'roll': 0},
    {'x': -54.622, 'y': 147.594, 'z': 12.8097, 'pitch': -23.424, 'yaw': -50.695, 'roll': 0},
    {'x': -87.981, 'y': 38.2766, 'z': 9.045, 'pitch': -9.5793, 'yaw': -120.583, 'roll': 0},
    {'x': -126.7717, 'y': -4.69908, 'z': 11.8058, 'pitch': -29.99356, 'yaw': 33.789333, 'roll': 0},
    {'x': 56.68508, 'y': 40.960907, 'z': 6.14143, 'pitch': -10.30762, 'yaw': -146.164, 'roll': 0},
]

STARTUP_WAIT = 30


class Runner:
    def __init__(self):
        self.cam_num = 0
        self.cameras = []          # list of (actor, queue, out_dir)
        self.world = None
        self.client = None
        self.tm = None
        self.walker_controllers = [] # <--- MODIFIED: To store walker controllers

    # Optional launcher if you want to start CarlaUE4.exe automatically
    def load_carla_env(self):
        print("Launching CARLA...")
        exe_dir = os.path.dirname(CARLA_EXE)
        proc = subprocess.Popen(
            [CARLA_EXE, "-windowed", "-quality-level=Epic"],
            cwd=exe_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"⌛ Waiting {STARTUP_WAIT}s for CARLA to start...")
        time.sleep(STARTUP_WAIT)
        return proc

    def set_weather(self):
        print("Changing the weather...")
        w = carla.WeatherParameters(
            cloudiness=weather_settings["cloudiness"],
            precipitation=weather_settings["precipitation"],
            precipitation_deposits=weather_settings["precipitation_deposits"],
            wind_intensity=weather_settings["wind_intensity"],
            fog_density=weather_settings["fog_density"],
            fog_distance=weather_settings["fog_distance"],
            wetness=weather_settings["wetness"],
            fog_falloff=weather_settings["fog_falloff"],
            scattering_intensity=weather_settings["scattering_intensity"],
            mie_scattering_scale=weather_settings["mie_scattering_scale"],
            rayleigh_scattering_scale=weather_settings["rayleigh_scattering_scale"],
            sun_altitude_angle=weather_settings["sun_altitude_angle"],
            sun_azimuth_angle=weather_settings["sun_azimuth_angle"],
        )
        self.world.set_weather(w)
        # A couple ticks to apply visuals deterministically
        for _ in range(2):
            self.world.tick()
            time.sleep(0.05)

    def generate_traffic(self, num_vehicles=60, num_walkers=30):
        """Spawn randomized vehicles and pedestrians; lights on; per-vehicle speed diffs."""
        print(f"🚗 Generating {num_vehicles} vehicles and {num_walkers} walkers...")

        bp_lib = self.world.get_blueprint_library()
        
        # <--- MODIFIED: All TM setup is removed from here. 
        # It's now done in run() *before* this function is called.
        # self.tm = self.client.get_trafficmanager()
        # self.tm.set_synchronous_mode(True)
        # self.tm.set_hybrid_physics_mode(True)
        # self.tm.global_percentage_speed_difference(0.0)
        # self.tm.set_random_device_seed(int(time.time()))

        # ---------------- Vehicles ----------------
        veh_bps = bp_lib.filter("vehicle.*")
        spawn_points = self.world.get_map().get_spawn_points()
        np.random.shuffle(spawn_points)

        vehicles = []
        for i in range(min(num_vehicles, len(spawn_points))):
            bp = np.random.choice(veh_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", np.random.choice(bp.get_attribute("color").recommended_values))
            if bp.has_attribute("driver_id"):
                bp.set_attribute("driver_id", np.random.choice(bp.get_attribute("driver_id").recommended_values))

            v = self.world.try_spawn_actor(bp, spawn_points[i])
            if not v:
                continue
            # Register with same TM instance/port (self.tm was set in run())
            v.set_autopilot(True, self.tm.get_port())

            # Per-vehicle speed variation (mostly slower), with some aggressive drivers
            self.tm.vehicle_percentage_speed_difference(v, np.random.uniform(0, 30))
            if np.random.random() < 0.10:
                self.tm.vehicle_percentage_speed_difference(v, np.random.uniform(-5, 0))

            if traffic_settings["car-lights-on"]:
                v.set_light_state(carla.VehicleLightState.All)

            vehicles.append(v)

        print(f"✅ Spawned {len(vehicles)} vehicles.")

        # ---------------- Walkers ----------------
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walkers = []
        for _ in range(num_walkers):
            nav_loc = self.world.get_random_location_from_navigation()
            if nav_loc is None:
                continue
            w = self.world.try_spawn_actor(np.random.choice(walker_bps), carla.Transform(nav_loc))
            if w:
                walkers.append(w)

        print(f"✅ Spawned {len(walkers)} walkers.")

        # Walker controllers
        self.walker_controllers = [] # <--- MODIFIED: Use class member
        ctrl_bp = bp_lib.find('controller.ai.walker')
        for w in walkers:
            c = self.world.try_spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            if c:
                c.start()
                dest = self.world.get_random_location_from_navigation()
                if dest:
                    c.go_to_location(dest)
                c.set_max_speed(np.random.uniform(1.0, 2.5))
                self.walker_controllers.append(c) # <--- MODIFIED: Add to class list

        print(f"🚶 Started {len(self.walker_controllers)} walker controllers.") # <--- MODIFIED

        # Warm-up ticks so TM & controllers actually begin moving
        print("⌛ Warming up simulation...")
        for _ in range(10):
            self.world.tick()
            time.sleep(0.02)

        print("🌆 Traffic generation complete.")

    def _safe_camera_transform(self, cfg):
        """Clamp/adjust camera position to a valid spot on current map and above the road."""
        loc = carla.Location(x=float(cfg['x']), y=float(cfg['y']), z=float(cfg['z']))
        rot = carla.Rotation(pitch=float(cfg['pitch']), yaw=float(cfg['yaw']), roll=float(cfg['roll']))

        # Project to road height if near any lane
        try:
            wp = self.world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Any)
            if wp is not None and loc.z < wp.transform.location.z + 5.0:
                loc.z = wp.transform.location.z + 15.0  # 15 m above road
        except Exception:
            pass

        return carla.Transform(loc, rot)

    def make_camera(self, config):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1920')
        bp.set_attribute('image_size_y', '1080')
        bp.set_attribute('fov', '90')

        cam_tf = self._safe_camera_transform(config)
        cam = self.world.spawn_actor(bp, cam_tf)

        bp_sem = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp_sem.set_attribute('image_size_x', '1920')
        bp_sem.set_attribute('image_size_y', '1080')
        bp_sem.set_attribute('fov', '90')
        cam_sem = self.world.spawn_actor(bp_sem, cam_tf)

        out_dir = f"DataGen/carla_captures/camera{self.cam_num}/raw"
        os.makedirs(out_dir, exist_ok=True)
        semantic_out_dir = f"DataGen/carla_captures/camera{self.cam_num}/semantic"
        os.makedirs(semantic_out_dir, exist_ok=True)
        self.cam_num += 1

        q = Queue()
        cam.listen(lambda img: q.put(("rgb", img, out_dir)))
        q_sem = Queue()  # added
        cam_sem.listen(lambda img: q_sem.put(("sem", img, semantic_out_dir)))  # added

        # store both in self.cameras
        self.cameras.append((cam, q, out_dir))
        self.cameras.append((cam_sem, q_sem, semantic_out_dir))


    def gather_metadata(self):
        data = {
            'close_call': ,
            'collision': ,
            
        }


    def run(self):
        # If you want to auto-launch CARLA, uncomment the next line and also terminate in finally
        # proc = self.load_carla_env()
        try:
            self.client = carla.Client("localhost", 2000)
            self.client.set_timeout(30.0)
            self.world = self.client.get_world()
            print(f"🗺️  Map: {self.world.get_map().name}")

            # --- Enable synchronous mode on world FIRST
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)

            # --- Then Traffic Manager sync
            # <--- MODIFIED: This is now the *only* place the TM is initialized
            self.tm = self.client.get_trafficmanager()
            self.tm.set_synchronous_mode(True)
            self.tm.set_hybrid_physics_mode(True)
            self.tm.global_percentage_speed_difference(0.0)
            self.tm.set_random_device_seed(int(time.time()))

            # Weather
            self.set_weather()

            # Traffic
            self.generate_traffic(
                num_vehicles=traffic_settings["number-of-vehicles"],
                num_walkers=traffic_settings["number-of-walkers"],
            )

            # Cameras
            for cam_cfg in camera_config:
                self.make_camera(cam_cfg)

            print("🎬 Recording… Press ESC to stop.")
            frame_idx = 0

            while not keyboard.is_pressed("esc"):
                frame = self.world.tick()

                # <--- MODIFIED: Add block to manage walkers
                for controller in self.walker_controllers:
                    if not controller.is_alive:
                        continue
                    
                    try:
                        # Check if walker has stopped (velocity is near zero)
                        walker_actor = controller.get_actor()
                        if walker_actor.get_velocity().length_squared() < 0.01:
                            # Give it a new random destination
                            new_dest = self.world.get_random_location_from_navigation()
                            if new_dest:
                                controller.go_to_location(new_dest)
                                controller.set_max_speed(np.random.uniform(1.0, 2.5))
                    except Exception:
                        pass # Actor might have been destroyed

                frame_idx += 1
                if frame_idx % 3 == 0:
                    for cam, q, out_dir in self.cameras:
                        try:
                            image = q.get(timeout=0.3)  # don’t let a single sensor stall ticks
                        except Empty:
                            continue

                        img_type, image, save_dir = image  # ✅ unpack tuple

                        if img_type == "rgb":
                            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                            arr = arr.reshape(image.height, image.width, 4)
                            rgb = arr[:, :, :3][:, :, ::-1]
                            fn = os.path.join(save_dir, f"frame_{image.frame:06d}.png")
                            Image.fromarray(rgb).save(fn)
                            print(f"Saved rgb image to {fn}")


                        elif img_type == "sem":
                            fn = os.path.join(save_dir, f"frame_{image.frame:06d}.png")
                            image.save_to_disk(fn, carla.ColorConverter.CityScapesPalette)
                            print(f"Saved segmented image to {fn}")
        finally:
            print("🧹 Cleaning up sensors and restoring settings...")
            for cam, _, _ in self.cameras:
                try:
                    cam.stop()
                    cam.destroy()
                except Exception:
                    pass

            # <--- MODIFIED: Add cleanup for walker controllers
            print("...Destroying walker controllers...")
            for controller in self.walker_controllers:
                try:
                    controller.stop()
                    controller.destroy()
                except Exception:
                    pass
            # <--- END OF MODIFIED BLOCK

            if self.world:
                try:
                    s = self.world.get_settings()
                    s.synchronous_mode = False
                    s.fixed_delta_seconds = None
                    self.tm.set_synchronous_mode(False) # Turn off TM sync first
                    self.world.apply_settings(s)
                except Exception:
                    pass

            # If you launched CARLA from this script, also terminate proc:
            # try:
            #     proc.terminate()
            # except Exception:
            #     pass

            print("✅ Done.")


if __name__ == "__main__":
    runner = Runner()
    runner.run()