checkpoint = ('https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/'
              'mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth')

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MobileNetV3',
        arch='large',        
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'),
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,
        in_channels=960,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(
            type='Normal', layer='Linear', mean=0., std=0.01, bias=0.),
        topk=(1, 5)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)