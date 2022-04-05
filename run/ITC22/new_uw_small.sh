cd ./src
python3 rl_test.py \
 --target LBIST \
 --exp_id ITC22_new_uw_small --pretrain False \
 --data_dir ../data/benchmarks/rl_test_large/ \
 --num_rounds 10 --dataset benchmarks \
 --gate_types INPUT,AND,NOT,BUFF \
 --dim_node_feature 4 --no_node_cop \
 --aggr_function aggnconv --wx_update \
 --bench_dir ../bench --no_labels \
 --save_bench --circuit_info --op --aig \
 --reward no_reward \
 --gpus 0 \
 --resume \
 --no_cp -1 --no_tp_each_round 10 \
 --RL_model non_level --no_cop_tpi \
 --ftpt update_with \
 --feature_pretrain_model ../exp/prob/pretrain/model_bak/model_30.pth \
 --ignore_action


