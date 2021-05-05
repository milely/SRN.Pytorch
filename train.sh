  CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_data_dir ../Data/test_lmdb/IIIT5K_3000 ../Data/test_lmdb/IIIT5K_3000 \
  --test_data_dir ../Data/test_lmdb/IIIT5K_3000 \
  --reuse_model '' \
  --lr 1e-4 \
  --batch_size 16 \
  --workers 2 \
  --height 64 \
  --width 256 \
  --voc_type LOWERCASE \
  --max_len 25 \

