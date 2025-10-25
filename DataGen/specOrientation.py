import carla
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)
 
world = client.get_world()
spectator = world.get_spectator()
transform = spectator.get_transform()
location = transform.location
rotation = transform.rotation
print(location)
print(rotation)