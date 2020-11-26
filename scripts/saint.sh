cd GraphSAINT
python -m graphsaint.tensorflow_version.train --data_prefix ./data/amazon --train_config ./train_config/table2/amazon_2_rw.yml --gpu -1

python -m graphsaint.tensorflow_version.train --data_prefix ./data/ppi --train_config ./train_config/table2/ppi2_rw.yml --gpu -1