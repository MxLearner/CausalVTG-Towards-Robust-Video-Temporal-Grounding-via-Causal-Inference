_base_ = ['models','datasets']

model = dict(
    type='NewVTG',
    dim_q = 512,   #  internvideo2_llama_text_feature 4096, clip_text_features 512
    dim_v = 512+2304, # internvideo2_videoclip_6b_w2s 768, clip_features 512, slowfast_features 2304
    hidden_dims=256,
    strides=(1, 2, 4, 8),
    buffer_size=1024,
    max_num_moment=50,
    adapter_cfg=dict(
        type='CausalAdapter',
        dropout=0.5,
        use_tef=True,
        video_cluster_path_list=[
            'features/Charades/cluster_clip_feat_256.npz',
            'features/Charades/cluster_slowfast_256.npz'
        ],
        query_cluster_path = 'features/Charades/cluster_clip_text_256.npz',
        num_clusters=256,
        pos_cfg=dict(
            type='PositionalEncoding',
            normalize=True,
            max_len=1024),
        tem_cfg=dict(
            type='TransformerDecoderLayer',
            heads=8,
            ratio=4,
            att_dropout=0.0,
            ffn_dropout=0.0,
            att_out_dropout=0.0,
            ffn_out_dropout=0.0,
            droppath=0.1,
            pre_norm=False,
            bias=True,
            norm_cfg=dict(type='LN'),
            act_cfg=dict(
                type='ReLU',
                inplace=True),
            order=('cross_att', 'self_att', 'ffn'),
            att_init_cfg=dict(
                type='xavier',
                distribution='uniform'),
            ffn_init_cfg=dict(type='kaiming'))),
    pyramid_cfg=dict(type='ConvPyramid'),
    pooling_cfg=dict(type='AdaPooling'),
    class_head_cfg=dict(
        type='ConvHead',
        kernal_size=3),
    coord_head_cfg=dict(
        type='ConvHead',
        kernal_size=3),
    loss_cfg=dict(
        type='BundleLoss',
        sample_radius=1.5,
        loss_cls=dict(
            type='FocalLoss',
            loss_weight=1.0),
        loss_reg=dict(
            type='L1Loss',
            loss_weight=0.2),
        loss_sal=dict(
            type='SampledNCELoss',
            loss_weight=0.01),
        loss_neg = dict(
            type='DynamicBCELoss',
            loss_weight=0.1)))

data = dict(
    train=dict(
        type='NewGrounding',
        label_path='data/charades/charades_sta_train_tvr_format.jsonl',
        video_path='',
        cache_path=['features/Charades/clip',
        'features/Charades/slowfast'
                    ],
        query_path='features/Charades/clip_text',
        min_video_len=5,
        fps=1,
        unit=0.1,
        loader=dict(
            batch_size=64, 
            num_workers=4,
            pin_memory=True,
            shuffle=True)),
    val=dict(
        type='NewGrounding',
        label_path='data/charades/charades_sta_test_rf.jsonl',
        video_path='',
        cache_path=['features/Charades/clip',
        'features/Charades/slowfast'
                    ],
        query_path='features/Charades/clip_text',
        fps=1,
        unit=0.1,
        loader=dict(
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False)))
stages = dict(
    epochs=50,
    optimizer=dict(
        type='AdamW',
        lr=0.00025,
        weight_decay=0.0001),
    lr_schedule=dict(
        type='epoch',
        policy='step',
        step=[30]),
    warmup=dict(
        type='iter',
        policy='linear',
        steps=500,
        ratio=0.001),
    grad_clip=dict(
        max_norm=35,
        norm_type=2),
    validation=dict(
        interval=1,
        # nms_cfg=dict(type='linear')
        ))
hooks = [
    dict(
        type='CheckpointHook',
        interval=1,
    ),
    dict(
        type='EvalHook',
        high_keys=[
            'MR-full-mAP', 'MR-full-mIoU', 'HL-min-VeryGood-mAP',
            'MR-full-R1@0.3', 'MR-full-R1@0.5', 'MR-full-R1@0.7', 'mAP','No-Answer-Acc'
        ]
    )
]