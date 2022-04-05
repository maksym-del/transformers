module load any/python/3.8.3-conda
module load cuda/11.3.1
source activate paper3

wandb login

export WANDB_PROJECT=multilingual_normformer
export WANDB_WATCH=all

BDIR=/gpfs/space/home/maksym95/third-paper
CDIR=/gpfs/space/projects/nlpgroup/hf_cache

cd $BDIR/transformers/examples/pytorch/language-modeling/

DATASET_PATH=$BDIR/saved_datasets/shuf-et_fr_lv-75000
TOKENIZER_PATH=$BDIR/saved_tokenizers/xlm-roberta-base-shuf-et_fr_lv-75000

RUN_NAME=scale_pre
OUT_DIR=$BDIR/saved_models/$RUN_NAME

LR=1e-4
BS=700
GACC=15
NUM_GPUS=2

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
    run_mlm.py \
        --model_type xlm-roberta \
        --dataset_path $DATASET_PATH \
        --tokenizer_name $TOKENIZER_PATH \
        --cache_dir $CDIR \
        --per_device_train_batch_size $BS \
        --per_device_eval_batch_size $BS \
        --max_seq_length 128 \
        --gradient_accumulation_steps $GACC \
        --do_train \
        --do_eval \
        --evaluation_strategy epoch \
        --num_train_epochs 200 \
        --logging_strategy steps \
        --logging_steps 1 \
        --logging_first_step \
        --save_strategy epoch \
        --logging_first_step \
        --save_steps 500000 \
        --seed 42 \
        --fp16 \
        --output_dir $OUT_DIR \
        --line_by_line \
        --pad_to_max_length \
        --learning_rate $LR \
        --config_overrides="scale_post=True,scale_pre=True,scale_fc=True,scale_attn=True,scale_heads=True,scale_resids=True" \
        --overwrite_output_dir \
        --report_to all \
        --run_name test_run \



# bsz steps lr ppl MNLI-m SST-2
# 256 1M 1e-4 3.99 84.7 92.7
# 2K 125K 7e-4 3.68 85.2 92.9
# 8K 31K 1e-3 3.77 84.6 92.8




