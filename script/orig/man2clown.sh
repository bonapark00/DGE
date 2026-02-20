python launch.py --config configs/dge.yaml --train --gpu 2 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn him into a clown" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.guidance.guidance_scale=12.5 \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"

