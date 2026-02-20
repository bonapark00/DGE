#!/bin/bash

python launch.py \
    --config configs/dge_clip-loss.yaml \
    --train --gpu 0 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn the man into a clown" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply" \
    system.seg_prompt="man" \
    system.target_prompt="clown" \
    system.mask_thres=0.6 \
    system.loss.lambda_d=100.0 \
    data.max_view_num=25 \
    data.max_edit_view_num=20 \
    system.camera_update_per_step=1500 \
    system.mask_update_at_step=-1 \
    name="clip-loss/wo-MaskUpdate/iter1/lambda_d100.0" &

sleep 3

python launch.py \
    --config configs/dge_clip-loss.yaml \
    --train --gpu 1 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn the man into a clown" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply" \
    system.seg_prompt="man" \
    system.target_prompt="clown" \
    system.mask_thres=0.6 \
    system.loss.lambda_d=150.0 \
    data.max_view_num=30 \
    data.max_edit_view_num=20 \
    system.camera_update_per_step=1500 \
    system.mask_update_at_step=-1 \
    name="clip-loss/wo-MaskUpdate/iter1/lambda_d150.0" &

sleep 3

python launch.py \
    --config configs/dge_clip-loss.yaml \
    --train --gpu 2 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn the man into a clown" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply" \
    system.seg_prompt="man" \
    system.target_prompt="clown" \
    system.mask_thres=0.6 \
    system.loss.lambda_d=200.0 \
    data.max_view_num=35 \
    data.max_edit_view_num=20 \
    system.camera_update_per_step=1500 \
    system.mask_update_at_step=-1 \
    name="clip-loss/wo-MaskUpdate/iter1/lambda_d200.0" &

sleep 3

python launch.py \
    --config configs/dge_clip-loss.yaml \
    --train --gpu 3 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn the man into a clown" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply" \
    system.seg_prompt="man" \
    system.target_prompt="clown" \
    system.mask_thres=0.6 \
    system.loss.lambda_d=180.0 \
    data.max_view_num=40 \
    data.max_edit_view_num=20 \
    system.camera_update_per_step=1500 \
    system.mask_update_at_step=-1 \
    name="clip-loss/wo-MaskUpdate/iter1/lambda_d180.0" &

wait