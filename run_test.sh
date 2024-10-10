# $1: --in_file_tst_dialog
# $2: filename of $1
# $3: --model_path
# $4: path for loading trained model
# $5: --ntask
# $6: the number of ntask

CUDA_VISIBLE_DEVICES="0" python3 ./main.py --mode test \
                                   --in_file_fashion ./data/mdata.wst.txt.2023.08.23 \
                                   --subWordEmb_path ./sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                   --model_file gAIa-final.pt \
                                   --mem_size 7 \
                                   --key_size 256 \
                                   --hops 3 \
                                   --eval_node [4096,2048,2048,2048,512][2048,2048,512] \
                                   --batch_size 100 \
                                   $1 $2 \
                                   $3 $4 \
                                   $5 $6 \
                                   $7 $8
