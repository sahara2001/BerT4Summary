python train.py \
    --data_dir data/processed_data\
    --bert_model pretrained_model\
    --GPU_index "0,2,3"\
    --train_batch_size 90\
    --num_train_epochs 400\
    --print_every 1000\
    --output_dir output

