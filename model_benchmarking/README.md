# Synthetic Data Generation for Traffic Intersections

### Environment Setup

```bash
conda create -n trafficsynth python=3.11 -y
conda activate trafficsynth

# install torch and torchvision with cuda 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# build mmcv first
pip install -U openmim
mim install mmengine
mim install mmcv==2.2.0

# install mmseg
git clone https://github.com/DRobinson4105/mmsegmentation.git
cd mmsegmentation
pip install -e .
cd ..

# install other dependencies while pinning torch==2.5.1 and torchvision==0.20.1
pip install -r requirements.txt -c constraints.txt
```

Notes:
- The `configs` directory is solely for incorporating trafficsynth into another mmsegmentation repository with the model configs for our benchmarking.
- The `models` directory only contains model architectures that were implemented, all other models must be tested with mmsegmentation.

## Running MMSegmentation

```bash
cd mmsegmentation
python tools/train {MODEL_CONFIG}
```

Tested config files:
- `configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py`
- `configs/gcnet/gcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py`
- `configs/sctnet/sctnet-b_seg50_8x2_160k_cityscapes.py`
- `configs/segman/segman_t_cityscapes.py`
- `configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py`
- `configs/segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512.py`