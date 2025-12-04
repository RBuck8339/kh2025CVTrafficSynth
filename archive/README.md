# CAP5415-Course Project Segmentation Benchmarks

## About

This is the code to run the segmentation model benchmarks in order to test our synthetic data generator. 

The models tested: **DDRNet** [repo](https://github.com/ydhongHIT/DDRNet)

Our synthetic dataset is generated using the Carla Simulator. Find the [docs](https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera) for reference on labels within segmented images.  



## Setup

### Installing Data
To install the data please run the following scripts:
```bash
./init-synthetic-data.sh 
```
To know it was successful you should see a data folder with carla_captures as one of your folders. This scripts fetches the latest pushed frames from the main branch.

To install the non-synthetic dataset run the following:
```bash
./init-real-data.sh
```
This will install the necessary data from the following [repo](https://math-ml-x.github.io/TrafficCAM/)

### Running the code

Requires Python3 and venv

```bash
python3 -m venv venv #create venv
source venv/bin/activate # start venv
pip3 install -r requirements.txt # install libraries

python3 main.py # to run source code
```