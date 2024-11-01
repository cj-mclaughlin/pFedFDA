# example launch script

# common arguments for all methods
BASE_ARGS="--num_clients 100 --sampling_prob 0.3 --local_epochs 5 --global_rounds 200 --eval_gap 200"

# data-scarcity arguments: e.g., train prop [0.25, 0.5, 0.75, 1.0]
# if we have created augmented datasets, we can also include --augmented to include covariate shift
DATA_ARGS="--train_prop 1.0"

# select model e.g., ['cnn', 'resnet18']
MODEL_ARGS="--model_name cnn"

# method-specific arguments
FEDAVG_ARGS="--method FedAvg"
FEDFDA_ARGS="--method pFedFDA"

# specify dataset arguments
DATASET_ARGS="--dataset cifar10 --num_classes 10"

# specify dataset partition arguments
PARTITION_ARGS="--partition_path cifar10_c100_dir05"

# FedAvg | FedAvgFT
python main.py ${BASE_ARGS} ${FEDAVG_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS} >> FedAvg.log

# pFedFDA
python main.py ${BASE_ARGS} ${FEDFDA_ARGS} ${DATASET_ARGS} ${PARTITION_ARGS} ${DATA_ARGS} ${MODEL_ARGS} >> pFedFDA_multi.log