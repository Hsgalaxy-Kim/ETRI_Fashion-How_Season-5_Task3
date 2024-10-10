# $1: --in_file_trn_dialog
# $2: filename of $1
# $3: --in_file_tst_dialog
# $4: filename of $3
# $5: --model_path
# $6: path for saving trained model
# $7: --model_file
# $8: filename of $7 (loaded file after task#1)
# $9: --ntask
# $10: the number of ntask

CUDA_VISIBLE_DEVICES="0" python3 ./main.py --mode train \
                                     --in_file_fashion ./data/mdata.wst.txt.2023.08.23 \
                                     --subWordEmb_path ./sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                     --mem_size 14 \
                                     --key_size 256 \
                                     --hops 3 \
                                     --eval_node [4096,2048,2048,2048,512][2048,2048,512] \
                                     --epochs 200 \
                                     --save_freq 5 \
                                     --batch_size 256 \
                                     --learning_rate 0.01 \
                                     --max_grad_norm 20.0 \
                                     --use_dropout True \
                                     --zero_prob 2e-2 \
                                     --permutation_iteration 3 \
                                     --num_augmentation 5 \
                                     --corr_thres 0.7 \
                                     $1 $2 \
                                     $3 $4 \
                                     $5 $6 \
                                     $7 $8 \
                                     $9 $10
