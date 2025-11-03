python launch.py \
    --config configs/dge_view5.yaml \
    --train --gpu 3 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn the man into an elf" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply" \
    system.seg_prompt="man" \
    system.target_prompt="an elf" \
    system.mask_thres=0.3 \
    system.loss.lambda_d=5
