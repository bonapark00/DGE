python launch.py \
    --config configs/dge_measure-time.yaml \
    --train --gpu 6 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn the man into a clown" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply" \
    system.seg_prompt="man" \
    system.mask_thres=0.6 \
    data.max_view_num=20 \
    system.camera_update_per_step=1500 \
    name="measure-time/wo-MaskUpdate/iter1/"


