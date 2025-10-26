Close Calls
===============

To Start, please refer to the below instructions for how to install Carla for Unreal Engine 4. 

<http://carla.readthedocs.io>


# What we did: 
CloseCalls uses the CARLA simulator in Unreal Engine 4 to generate realistic traffic intersection data. Every scene is automatically labeled with bounding boxes, object attributes, and actions. Expanding on traditional tasks such as predicting vehicle speed or location, we add the capability to detect car crashes or scenarios in which a car nearly hits a pedestrian/bicyclist. This provides a way to simulate vehicle-vehicle and vehicle-human interactions without compromising real people. This information can be used for city planning, determining where in a city may be dangerous and in need of increased development for pedestrian safety. After generating data in various scenes such as clear-sunny days, a foggy morning, and a night, we employ a OpenTAD Machine Learning Model to train and test using our synthetic datasets using the AMD Developer Cloud.

# How to use it:
We included a comprehensive runner script in `DataGen/runner.py` that will open UE4 and simulate an environment as dictated by `traffic_settings` and `weather_settings`. This will generate data for (currently 8) traffic cameras and store them in `DataGen/carla_captures/{scene}/camera_{camera_num}/` for each camera

You can then run this using the files seen in the Predictor directory

## How to use the Data Analyst Helper
cd into the gemini_helper folder and run the `gemini_prompt.py` and add the csv file. Then chat with the LLM and gather insights!


# Made by
Ronan Buck, Ahmet Cetin, Yussef Zahrani
