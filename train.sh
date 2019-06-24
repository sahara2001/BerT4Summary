python train.py \
    --data_dir data/processed_data\
    --bert_model ernie\
    --GPU_index "0,2"\
    --train_batch_size 50\
    --num_train_epochs 20\
    --print_every 1000\
    --output_dir output

