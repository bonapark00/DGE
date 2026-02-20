# 여기 argument들이 config/dge.yaml 파일에 들어가는 것들임
python launch.py --config configs/dge.yaml --train --gpu 2 \
    trainer.max_steps=1500 \
    system.prompt_processor.prompt="Turn him into a spider man with Mask" \
    data.source="/working/style-transfer/VcEdit/gs_data/face/" \
    system.gs_source="/working/style-transfer/VcEdit/gs_data/trained_gs_models/face/point_cloud.ply" \
    data.max_view_num=5  \