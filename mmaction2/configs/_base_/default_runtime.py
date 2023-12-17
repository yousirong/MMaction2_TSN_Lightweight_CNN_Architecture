default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
# load_from = './abnormal_model.pth'
# resume_from= "./work_dirs/tsn_moblieone_s0_1x1x8_20e_Dassult/last_checkpoint"
# resume_from= "./work_dirs/tsn_moblieone_s1_1x1x8_20e_Dassult/last_checkpoint"
# resume_from= "./work_dirs/tsn_moblieone_s2_1x1x8_20e_Dassult/last_checkpoint"
# resume_from= "./work_dirs/tsn_moblieone_s3_1x1x8_20e_Dassult/last_checkpoint"
# resume_from= "./work_dirs/tsn_moblieone_s4_1x1x8_20e_Dassult/last_checkpoint"
# resume_from= "./work_dirs/tsn_r50_1x1x8_20e_Dassult/last_checkpoint"
# resume_from= "./work_dirs/tsn_r50_1x1x8_20e_Dassult_aihub/last_checkpoint"
# resume = True
resume=False
