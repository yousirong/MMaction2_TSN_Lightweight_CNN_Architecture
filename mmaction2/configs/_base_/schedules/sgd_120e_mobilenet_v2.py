train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=120, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
# val_cfg = dict(type='EpochBasedValLoop', max_epochs=100)
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=120,
        by_epoch=True,
        milestones=[40, 80],
        gamma=0.98)
]

# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
#     clip_grad=dict(max_norm=40, norm_type=2))

# learning policy
# param_scheduler = dict(type='StepLR', by_epoch=True, step_size=1, gamma=0.98)
# learning policy
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004),
    clip_grad=dict(max_norm=40, norm_type=2))


auto_scale_lr = dict(base_batch_size=8)

