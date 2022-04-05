cd ./src
python3 rl_test_ATPG_PC.py \
 --target ATPG_PC \
 --exp_id rl_test --pretrain False \
 --data_dir ../data/benchmarks/small_aig/ \
 --num_rounds 10 --dataset benchmarks \
 --gate_types INPUT,AND,NOT,BUFF \
 --dim_node_feature 4 --no_node_cop \
 --aggr_function aggnconv --wx_update \
 --bench_dir ../bench --no_labels \
 --circuit_info --op --aig \
 --reward no_reward \
 --resume --ignore_action \
 --no_cp 100 --no_tp_each_round 1 \
 --RL_model non_level \
 --ftpt update_with \
 --feature_pretrain_model ../exp/prob/pretrain/model_bak/model_30.pth

