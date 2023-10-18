SEED=$(($RANDOM << 15 | $RANDOM))
PORT=$((25090 + $RANDOM % 1024))
deepspeed --hostfile=hostfile --master_port $PORT main.py --seed $SEED\
    --recipe-pth ./recipe/vit-B=4096-localadamw-cos.yml\
    --model vit_base\
    --total-batch-size 4096 --physical-batch-size 128 --optimizer localadamw\
    --epochs 300 --epochs-for-sche 300 --warmup-steps 10000 --log-per-step 1 --steps-per-epoch 312\
    --init-h 4 --min-h 4 --alpha 0 --power 2\
    --avg-m 0 --avg-v 0\
    --scheduler cosine\
    --save-freq 20\
    --wd 0.05 --max-lr 0.008 --debug 1\
    --strong-aug 1

