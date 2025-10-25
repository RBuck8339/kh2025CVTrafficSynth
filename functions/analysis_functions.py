import pandas as pd
import json

def carla_dict_to_df(metadata_list):
    flattened_data = []

    for entry in metadata_list:
            flat_entry = {
            'timestamp': entry.get('timestamp'),
            'vehicle_id': entry.get('vehicle_id'),
            'type': entry.get('type'),
            'location_x': entry.get('location', {}).get('x'),
            'location_y': entry.get('location', {}).get('y'),
            'location_z': entry.get('location', {}).get('z'),
            'velocity_x': entry.get('velocity', {}).get('x'),
            'velocity_y': entry.get('velocity', {}).get('y'),
            'velocity_z': entry.get('velocity', {}).get('z'),
            'speed_kmh': entry.get('speed_kmh'),
            'road_id': entry.get('road_id'),
            'lane_id': entry.get('lane_id'),
            'traffic_light_state': entry.get('traffic_light_state'),
            'weather': entry.get('weather')
        }
    flattened_data.append(flat_entry)

    df = pd.DataFrame(flattened_data)

    return df

if __name__ == "__main__":
    sample_data = {}

    df = carla_dict_to_df(sample_data)

    print("CARLA Metadata DataFrame:")
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")

    df.to_csv('carla_traffic_data.csv', index=False)
    print("\nData saved to 'carla_traffic_data.csv'")
