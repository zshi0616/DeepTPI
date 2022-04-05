cd ./src
python3 rl_train.py \
 --exp_id nonpt_rand \
 --pretrain False \
 --data_dir ../data/benchmarks/random/ \
 --num_rounds 10 --dataset benchmarks \
 --no_labels \
 --gate_types INPUT,AND,NOT,BUFF --dim_node_feature 4 --no_node_cop \
 --aggr_function aggnconv --wx_update \
 --reward sparse --lr 0.0001 --op --aig

