EXP_ID='recgnn_gatedsum'
ARRG='gated_sum' # ['deepset', 'aggnconv', 'gated_sum', 'conv_sum']
ARCH='recgnn' # ['recgnn', 'convgnn']
cd src
python test.py prob --exp_id $EXP_ID --data_dir ../data/benchmarks/merged/ --num_rounds 10 --dataset benchmarks --gpus -1 --batch_size 1 --gate_types INPUT,AND,NOT --dim_node_feature 3 --no_node_cop --wx_update --aggr_function $ARRG --arch $ARCH --load_model model_best.pth --test_num_rounds 10 #--un_directed #--debug 1