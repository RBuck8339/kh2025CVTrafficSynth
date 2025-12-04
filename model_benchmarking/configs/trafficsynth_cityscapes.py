_base_ = ['./cityscapes_1024x1024.py']

dataset_type = 'TrafficSynthDataset'
data_root_traffic = 'data/trafficsynth_cityscapes_train'
crop_size = (512, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_traffic,
        data_prefix=dict(
            img_path='images',
            seg_map_path='segmentation'),
        pipeline=train_pipeline,
    ),
)
