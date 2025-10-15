python launch.py --config configs/dge.yaml --train --gpu 0 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn him into a spider man with Mask" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply"