#!bin/bash

python main.py \
    --config-path /DMNSP/configs/class \
    --config-name cifar100_10-10.yaml \
    dataset_root="/data/**/" \
    class_order="/DMNSP/class_orders/cifar100.yaml"


