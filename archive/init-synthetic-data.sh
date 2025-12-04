echo "Segmentation Init. Going to installing synthetic data"

# data init
mkdir data
cd data

# installing TrafficSynth data
# Source: https://github.com/RBuck8339/kh2025CVTrafficSynth/tree/main/DataGen/carla_captures/
repo="https://github.com/RBuck8339/kh2025CVTrafficSynth/archive/refs/heads/main.zip"

# fetch data
wget $repo
unzip main.zip

# copy over our data
cp -r kh2025CVTrafficSynth-main/DataGen/carla_captures . 

# clean up
rm main.zip
rm -rf kh2025CVTrafficSynth-main
cd .. # return up
