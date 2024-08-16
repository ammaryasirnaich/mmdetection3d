export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node 2 val.py --config configs/sparseocc_r50_nuimg_704x256_8f.py --weights pretrain/sparseocc_r50_nuimg_704x256_8f.pth