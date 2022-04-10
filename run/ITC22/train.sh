cd ./src

python3 rl_train.py \
 --exp_id ITC22_deeptpi_train \
 --RL_model non_level \
 --ftpt update_with \
 --feature_pretrain_model ../exp/deepgate/pretrained.pth \
 --target LBIST \
 --data_dir ../data/ITC22_dataset/train/ \
 --num_rounds 10 --dataset benchmarks \
 --no_labels \
 --gate_types INPUT,AND,NOT,BUFF --dim_node_feature 4 --no_node_cop \
 --aggr_function aggnconv --wx_update \
 --reward sparse --lr 0.0001 --op --aig \

