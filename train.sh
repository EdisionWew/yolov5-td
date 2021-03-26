#python3 -m torch.distributed.launch \
#        --nproc_per_node=4 \
#        --master_port=$((RANDOM + 20000)) \
#        ./train.py \
#        --weights ./runs/train/exp35/weights/best.pt \
#        --data ./data/logo_v0.4.yaml

#  ADS logo
python3 -m torch.distributed.launch \
        --nproc_per_node=3 \
        --master_port=$((RANDOM + 20000)) \
        ./train.py \
        --batch-size 150 \
        --data ./data/supreme_test.yaml
