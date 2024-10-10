### train task#1 ###
sh run_train.sh --in_file_trn_dialog ./data/filtered_output1.txt --in_file_tst_dialog ./data/cl_eval_task1.wst.dev --model_path ./model
### test task#1 ###
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task1.wst.dev --model_path ./model
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task2.wst.dev --model_path ./model
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task3.wst.dev --model_path ./model
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task4.wst.dev --model_path ./model
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task5.wst.dev --model_path ./model
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task6.wst.dev --model_path ./model

### train task#2 ###
sh run_train.sh --in_file_trn_dialog ./data/filtered_output2.txt --in_file_tst_dialog ./data/cl_eval_task2.wst.dev --model_path ./model --model_file gAIa-final.pt --ntask 2
### test task#2 ###
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task1.wst.dev --model_path ./model --ntask 2
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task2.wst.dev --model_path ./model --ntask 2
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task3.wst.dev --model_path ./model --ntask 2
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task4.wst.dev --model_path ./model --ntask 2
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task5.wst.dev --model_path ./model --ntask 2
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task6.wst.dev --model_path ./model --ntask 2

### train task#3 ###
sh run_train.sh --in_file_trn_dialog ./data/filtered_output3.txt  --in_file_tst_dialog ./data/cl_eval_task3.wst.dev --model_path ./model --model_file gAIa-final.pt --ntask 3
### test task#3 ###
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task1.wst.dev --model_path ./model --ntask 3
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task2.wst.dev --model_path ./model --ntask 3
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task3.wst.dev --model_path ./model --ntask 3
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task4.wst.dev --model_path ./model --ntask 3
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task5.wst.dev --model_path ./model --ntask 3
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task6.wst.dev --model_path ./model --ntask 3

### train task#4 ###
sh run_train.sh --in_file_trn_dialog ./data/filtered_output4.txt  --in_file_tst_dialog ./data/cl_eval_task4.wst.dev --model_path ./model --model_file gAIa-final.pt --ntask 4
### test task#4 ###
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task1.wst.dev --model_path ./model --ntask 4
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task2.wst.dev --model_path ./model --ntask 4
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task3.wst.dev --model_path ./model --ntask 4
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task4.wst.dev --model_path ./model --ntask 4
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task5.wst.dev --model_path ./model --ntask 4
# sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task6.wst.dev --model_path ./model --ntask 4

# ### train task#5 ###
sh run_train.sh --in_file_trn_dialog ./data/filtered_output5.txt  --in_file_tst_dialog ./data/cl_eval_task5.wst.dev --model_path ./model --model_file gAIa-final.pt --ntask 5
# ### test task#5 ###
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task1.wst.dev --model_path ./model --ntask 5
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task2.wst.dev --model_path ./model --ntask 5
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task3.wst.dev --model_path ./model --ntask 5
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task4.wst.dev --model_path ./model --ntask 5
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task5.wst.dev --model_path ./model --ntask 5
# # sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task6.wst.dev --model_path ./model --ntask 5

# ### train task#6 ###
sh run_train.sh --in_file_trn_dialog ./data/filtered_output6.txt  --in_file_tst_dialog ./data/cl_eval_task6.wst.dev --model_p/ath ./model --model_file gAIa-final.pt --ntask 6
# ### test task#6 ###
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task1.wst.dev --model_path ./model --ntask 6
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task2.wst.dev --model_path ./model --ntask 6
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task3.wst.dev --model_path ./model --ntask 6
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task4.wst.dev --model_path ./model --ntask 6
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task5.wst.dev --model_path ./model --ntask 6
sh run_test.sh --in_file_tst_dialog ./data/cl_eval_task6.wst.dev --model_path ./model --ntask 6 --inference True

