_base_ = [
    '../../../_base_/models/tsn_mobileone_s0_videotest.py', '../../../_base_/schedules/sgd_120e.py',
    '../../../_base_/default_runtime.py'
]
# _base_설명 :tsn_mobileone_s0 backbone 모델 사용, schedules/adam 20epoch optimizer 사용
# !pip install timm 
# !pip install mmpretrain  

# dataset settings
dataset_type = 'VideoDataset' 
data_root = ''
data_root_val = ''
ann_file_train = './datasets/allData/Youtube_abnormal.txt'
ann_file_val = './datasets/allData/Youtube_abnormal.txt'
ann_file_test = './datasets/allData/Youtube_abnormal.txt'

# dataset_type = 'RawframeDataset' 
# data_root = ''
# data_root_val = ''
# ann_file_train = './datasets/allData/TL_train.txt'
# ann_file_val = './datasets/allData/TL_val.txt'
# ann_file_test = './datasets/allData/TL_test.txt'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=5, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=5,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
# test_pipeline = [
#     dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=1,
#         frame_interval=1,
#         num_clips=8,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='CenterCrop', crop_size=(960, 720)),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='PackActionInputs'),
# ]
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# train_pipeline = [
#     dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
#     dict(type= 'RawFrameDecode'),
#     dict(
#         type='MultiScaleCrop',
#         input_size=224,
#         scales=(1, 0.875, 0.75, 0.66),
#         random_crop=False,
#         max_wh_scale_gap=1),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='PackActionInputs'),
# ]
# val_pipeline = [
#     dict(
#         type='SampleFrames',
#         clip_len=1,
#         frame_interval=1,
#         num_clips=8,
#         test_mode=True),
#     dict(type= 'RawFrameDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='PackActionInputs'),
# ]
# # test_pipeline = [
# #     dict(
# #         type='SampleFrames',
# #         clip_len=1,
# #         frame_interval=1,
# #         num_clips=8,
# #         test_mode=True),
# #     dict(type= 'RawFrameDecode'),
# #     dict(type='Resize', scale=(-1, 256)),
# #     dict(type='TenCrop', crop_size=224),
# #     dict(type='FormatShape', input_format='NCHW'),
# #     dict(type='PackActionInputs')
# # ]
# test_pipeline = [
#     dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=1,
#         frame_interval=1,
#         num_clips=8,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='PackActionInputs')
# ]
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    # sampler=dict(type='CustomSampler', shuffle=True, ann_file=ann_file_train),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        # modality='RGB',
        modality='Flow', # default RGB
        # data_prefix=dict(img=data_root), 
        data_prefix=dict(video=data_root), 
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    # sampler=dict(type='CustomSampler', shuffle=False, ann_file=ann_file_val),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        # modality='RGB',
        modality='Flow', # default RGB
        # data_prefix=dict(img=data_root_val),
        data_prefix=dict(video=data_root_val), 
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), 
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        # modality='RGB',
        modality='Flow', # default RGB
        # data_prefix=dict(img=data_root_val),
        data_prefix=dict(video=data_root_val), 
        pipeline=test_pipeline,
        test_mode=True))
custom_hooks = [
    dict(
        type='EMAHook',
        momentum=5e-4,
        priority='ABOVE_NORMAL',
        update_buffers=True)
]
# optim_wrapper = dict(paramwise_cfg=dict(norm_decay_mult=0.))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator
#ActionVisualizer
visualizer = dict(  
    type='ActionVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)
# runtime settings
work_dir = './work_dirs/tsn_moblieone_s0_1x1x8_20e_Dassult/'

