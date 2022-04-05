cd src
python3 pretrain.py --exp_id pretrain \
 --pretrain 1 \
 --data_dir ../data/benchmarks/pretrain/ \
 --num_rounds 10 --dataset benchmarks --gpus 0 \
 --gate_types INPUT,AND,NAND,OR,NOR,NOT,XOR \
 --dim_node_feature 7 \
 --no_node_cop --aggr_function aggnconv --wx_update \
 --lr 0.01
 --resume
