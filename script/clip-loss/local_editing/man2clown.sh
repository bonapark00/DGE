python launch.py \
    --config configs/dge_view5.yaml \
    --train --gpu 7 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn the man into a clown" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply" \
    system.seg_prompt="man" \
    system.target_prompt="clown" \
    system.mask_thres=0.6 \
    system.loss.lambda_d=0.0 \
    data.max_view_num=20 \
    system.camera_update_per_step=500 \
    system.mask_update_at_step=-1 \
    name="clip-loss/wo-MaskUpdate/iter3/lambda_d0.0"