# This script runs model training

python3 ./train.py \
    --dataset "$DATASET_PATH" \
    --domain_id "domain_id" \
    --content_id "content" \
    --label_id "article_reads" \
    --embedding_id "bert_embedding" \
    --base_model "bert-base-multilingual-cased" \
    --test_size 0.25 \
    --samples_per_domain 10 \
    --gcn_embed_dims 256 128 128 128 64 \
    --gcn_layer "SAGEConv" \
    --gcn_act "relu" \
    --epochs 300 \
    --learning_rate 0.0001 \
    --optimizer "AdamW" \
    --device "cuda" \
