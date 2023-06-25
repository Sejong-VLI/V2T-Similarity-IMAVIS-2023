DATATYPE=msvd
DATA_PATH=./dataset/MSVD
FEATURES_PATH=./features/MSVD_Clip4Clip_features_vit16.pickle
INIT_MODEL=CLIP4Caption/weight/univl.pretrained.bin
OUTPUT_ROOT=ckpts_msvd
LEARNING_RATE=(1e-5)

MODEL_FILE=CLIP4Caption/weight/univl.pretrained.bin
MODEL_FILE_RET=./pretrained/MSVD/pytorch_model.bin.0

LAMDA=(1.0)
GAMMA_RL=(0.5)


declare -a hid_layers=(
    "2 2"
)

for elem in "${hid_layers[@]}"; do
  read -a hid_layer <<< "$elem"  # uses default whitespace IFS
  for lr in "${LEARNING_RATE[@]}"
  do
  for gamma in "${GAMMA_RL[@]}"
  do
    for ret_weight in "${LAMDA[@]}"
    do
        python -m torch.distributed.launch --nproc_per_node=2 \
        train.py --do_train --num_thread_reader=8\
        --epochs=100 --batch_size=256 --n_display=100 --gradient_accumulation_steps 1\
        --data_path ${DATA_PATH} --features_path ${FEATURES_PATH} --patience 150 \
        --output_dir ${OUTPUT_ROOT}/vit16_ckpt_${DATATYPE}_reinforce${gamma}_rw${ret_weight}_lr_${lr}_vl${hid_layer[0]}_dl${hid_layer[1]}_${clip}_seed${seed} \
        --output_dir_retrieval ${OUTPUT_ROOT}/ckpt_msvd_retrieval2_lr${LEARNING_RATE}_vl${hid_layer[0]}_dl${hid_layer[1]} \
        --bert_model bert-base-uncased --do_lower_case \
        --lr ${lr} --max_words 48 --max_frames 20 --batch_size_val 32 \
        --visual_num_hidden_layers ${hid_layer[0]} --decoder_num_hidden_layers ${hid_layer[1]} \
        --datatype ${DATATYPE} --init_model ${INIT_MODEL} --video_dim 512 \
        --gamma_rl ${gamma} --reward_multiplier 0.1 --retrieval_weight ${ret_weight} \
        --model_file ${MODEL_FILE} --model_file_retrieval ${MODEL_FILE_RET} \
        --eval_type captioning --sim_header meanP --loose_type --freeze_layer_num 0 --linear_patch 2d
    done
    done
  done
done
