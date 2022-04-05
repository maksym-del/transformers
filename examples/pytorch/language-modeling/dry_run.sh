
module load any/python/3.8.3-conda
module load cuda/11.3.1
source activate paper3


BDIR=/gpfs/space/home/maksym95/third-paper
CDIR=/gpfs/space/projects/nlpgroup/hf_cache

cd $BDIR/transformers/examples/pytorch/language-modeling/

DATASET_PATH=$BDIR/saved_datasets/shuf-et_fr_lv-75000
TOKENIZER_PATH=$BDIR/saved_tokenizers/xlm-roberta-base-shuf-et_fr_lv-75000

python run_mlm.py \
    --model_type xlm-roberta \
    --dataset_path $DATASET_PATH \
    --tokenizer_name $TOKENIZER_PATH \
    --cache_dir $CDIR \
    --per_device_train_batch_size 100 \
    --per_device_eval_batch_size 100 \
    --max_seq_length 128 \
    --gradient_accumulation_steps 10 \
    --do_train \
    --do_eval \
    --output_dir $BDIR/saved_models/scale_post \
    --line_by_line \
    --pad_to_max_length \
    --learning_rate 0.0005 \
    --config_overrides="scale_pre=True,scale_post=False" \
    --overwrite_output_dir



# bsz steps lr ppl MNLI-m SST-2
# 256 1M 1e-4 3.99 84.7 92.7
# 2K 125K 7e-4 3.68 85.2 92.9
# 8K 31K 1e-3 3.77 84.6 92.8




