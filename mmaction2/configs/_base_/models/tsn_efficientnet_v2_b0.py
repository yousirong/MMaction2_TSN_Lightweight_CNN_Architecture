checkpoint = ('https://download.openmmlab.com/mmclassification/v0/efficientnetv2/'
              'efficientnetv2-b0_3rdparty_in1k_20221221-9ef6e736.pth')

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='mmpretrain.EfficientNetV2',
        arch='b0',        
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'),
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,
        in_channels=1280,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.2,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)