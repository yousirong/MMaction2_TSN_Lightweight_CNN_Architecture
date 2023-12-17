checkpoint = ('https://download.openmmlab.com/mmclassification/'
              'v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth')
# model settings

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ShuffleNetV2TSN', 
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'),
        norm_eval=False,
        widen_factor=1.0),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,
        in_channels=1024,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob',
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),),
    
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)


