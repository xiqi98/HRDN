CUDA_VISIBLE_DEVICES=1 python NTIRE_val_tta.py \
--val_batch_size=1 \
--num_workers=8 \
--input_root='test' \
--output_root='test_final' \
--pretrained_model='checkpoint/hrnet_se_final.pth'
