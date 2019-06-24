export CUDA_VISIBLE_DEVICES=0

python predict.py \
    --model_path output/*19:42*/BertAbsSum_19.bin\
    --config_path output/*19:42*/config.json\
    --eval_path data/processed_data/eval.csv\
    --bert_model ernie\
    --result_path result
