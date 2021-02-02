python3 -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=$((RANDOM + 20000)) \
        ./train.py \
        --weights ./yolov5m.pt \
        --data ./data/logo_total.yaml
