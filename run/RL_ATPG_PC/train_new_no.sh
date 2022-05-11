cd ./src

python3 rl_train.py \
 --exp_id atpg_pc_new_no \
 --RL_mode train \
 --RL_model non_level \
 --ftpt no \
 --target ATPG_PC \
 --data_dir ../data/benchmarks/iscas_aig/ \
 --num_rounds 10 --dataset benchmarks \
 --no_labels \
 --gate_types INPUT,AND,NOT,BUFF --dim_node_feature 4 --no_node_cop \
 --aggr_function aggnconv --wx_update \
 --reward cont --op --aig \
 --RL_max_times 400 \
 --lr 1e-3 \
 --resume

