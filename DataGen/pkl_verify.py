import pickle
import pprint

with open("DataGen/carla_captures/Sunny/camera2/metadata/30.pkl", "rb") as f:
    data = pickle.load(f)

pprint.pprint(data, width=120)