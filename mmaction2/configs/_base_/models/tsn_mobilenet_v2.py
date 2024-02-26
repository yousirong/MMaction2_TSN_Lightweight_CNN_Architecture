checkpoint = ('https://download.openmmlab.com/mmclassification/v0/'
              'mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth')

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'),
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,
        in_channels=1280,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob',
        topk=(1, 5)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)