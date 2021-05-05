  CUDA_VISIBLE_DEVICES=0 python eval.py \
  --test_data_dir ../Data/test_lmdb/ \
  --reuse_model ./ckpt/SRN_best.pth \
  --lr 1e-4 \
  --workers 2 \
  --height 64 \
  --width 256 \
  --voc_type LOWERCASE \
  --max_len 25 \
