cd ./src
python3 pretrain.py \
 --exp_id pretrain --pretrain True \
 --data_dir ../data/benchmarks/dac22_dataset/ \
 --num_rounds 10 --dataset benchmarks \
 --gate_types INPUT,AND,NOT,BUFF \
 --dim_node_feature 4 --no_node_cop \
 --aggr_function aggnconv --wx_update --lr 0.0001 \
 --resume







