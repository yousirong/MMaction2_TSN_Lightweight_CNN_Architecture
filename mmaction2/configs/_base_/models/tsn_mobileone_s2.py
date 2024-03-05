checkpoint = ('https://download.openmmlab.com/mmclassification/'
              'v0/mobileone/mobileone-s2_8xb32_in1k_20221110-9c7ecb97.pth')
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='mmpretrain.MobileOne',
        arch='s2',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'),
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,
        in_channels=2048,
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
