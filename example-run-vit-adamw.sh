SEED=$(($RANDOM << 15 | $RANDOM))
PORT=$((25090 + $RANDOM % 1024))
deepspeed --master_port $PORT --hostfile=hostfile main.py --seed $SEED\
    --recipe-pth ./recipe/vit-B=4096-adamw-cos.yml\
    --model vit_base\
    --total-batch-size 4096\
    --max-lr 0.008 --wd 0.1\
    --epochs 300 --epochs-for-sche 300 --warmup 1 --warmup-steps 10000 --log-per-step 1 --steps-per-epoch 312\
    --init-h 1 --scheduler cosine --save-freq 30 --strong-aug 1 --gradient-clipping 1 --debug 1\