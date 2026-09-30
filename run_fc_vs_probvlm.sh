#!/bin/bash
# pFedFDA with frozen CLIP: FC-only vs. MLP adapter vs. ProbVLM adapter training (CLIP encoder is never trained/aggregated/saved).
# All runs train on the first 50 of 100 clients (NUM_TRAIN_CLIENTS in main.py), then evaluate the training clients,
# the 50 held-out clients, and CIFAR-C corrupted versions of the held-out clients.
# Checkpoints, eval CSVs, and logs go to results/<exp_name>/ (distinct per run).
set -eo pipefail

# common arguments for all runs
BASE_ARGS="--method pFedFDA --num_clients 100 --sampling_prob 0.3 --global_rounds 50 --eval_gap 50 --batch_size 16 --no-from_checkpoint"
DATASET_ARGS="--dataset cifar10 --num_classes 10 --partition_path cifar10_c100_dir05"
# local epochs for the runs with a trainable adapter (same budget for MLP adapter and ProbVLM)
ADAPTER_EPOCHS=1

# FC-only: frozen CLIP features, FDA head only
FC_EXP="clip_fc_train50of100"
mkdir -p results/${FC_EXP}
python main.py ${BASE_ARGS} ${DATASET_ARGS} --model_name clip --local_epochs 1 \
    --exp_name ${FC_EXP} --checkpoint_path results/${FC_EXP}/checkpoint_latest.pt \
    2>&1 | tee results/${FC_EXP}/train.log

# MLP adapter: frozen CLIP + residual MLP (512 -> 256 -> 512) trained with cross-entropy through the FDA head
ADAPTER_EXP="adapter_train50of100"
mkdir -p results/${ADAPTER_EXP}
python main.py ${BASE_ARGS} ${DATASET_ARGS} --model_name adapter --local_epochs ${ADAPTER_EPOCHS} \
    --exp_name ${ADAPTER_EXP} --checkpoint_path results/${ADAPTER_EXP}/checkpoint_latest.pt \
    2>&1 | tee results/${ADAPTER_EXP}/train.log

# ProbVLM: frozen CLIP + ProbVLM adapter trained locally, FDA head on the adapter means
PROBVLM_EXP="probvlm_train50of100"
mkdir -p results/${PROBVLM_EXP}
python main.py ${BASE_ARGS} ${DATASET_ARGS} --model_name probvlm --local_epochs ${ADAPTER_EPOCHS} --probvlm_lr 1e-3 \
    --exp_name ${PROBVLM_EXP} --checkpoint_path results/${PROBVLM_EXP}/checkpoint_latest.pt \
    2>&1 | tee results/${PROBVLM_EXP}/train.log
