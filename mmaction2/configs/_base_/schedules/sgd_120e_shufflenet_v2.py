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
        gamma=0.1)
]

# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
#     clip_grad=dict(max_norm=40, norm_type=2))
# learning policy
# param_scheduler = [
#     dict(type='ConstantLR', factor=0.1, by_epoch=True, begin=0, end=120),
#     dict(type='PolyLR', eta_min=0, by_epoch=True, begin=120)
# ]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(norm_decay_mult=0),
    clip_grad=dict(max_norm=40, norm_type=2)
)

