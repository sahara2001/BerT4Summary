export CUDA_VISIBLE_DEVICES=0

python predict.py \
    --model_path output/*14:59*/BertAbsSum_4.bin\
    --config_path output/*14:59*/config.json\
    --eval_path data/processed_data/eval.csv\
    --bert_model pretrained_model\
    --result_path result
