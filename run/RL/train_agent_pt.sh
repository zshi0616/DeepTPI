cd ./src
python3 rl_train.py \
 --exp_id rl_pre \
 --data_dir ../data/random_circuits/random_aig_50/ \
 --num_rounds 10 --dataset benchmarks \
 --no_labels \
 --gate_types INPUT,AND,NOT --dim_node_feature 3 --no_node_cop \
 --aggr_function aggnconv --wx_update --lr 0.00001 --resume

