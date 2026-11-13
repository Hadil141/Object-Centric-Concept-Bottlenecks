#!/bin/bash

python train.py \
    --dataset coco \
    --task_type multilabel \
    --num_epochs 20 \
    --use_object_concepts \
    --min_object_size 0.01 \
    --max_object_size 0.85 \
    --learning_rate 0.01 \
    --min_score 0.2 \
    --aggregation sum \
    --num_objects_training 7 \
    --seed 42 \
    --prediction_level super \
    --object_detection_model mask-rcnn

python train.py \
    --dataset coco \
    --data_dir datasets/coco \
    --task_type multilabel \
    --num_epochs 20 \
    --use_object_concepts \
    --num_objects 7 \
    --min_object_size 0.01 \
    --max_object_size 0.85 \
    --learning_rate 0.001 \
    --min_score 0.2 \
    --aggregation sum \
    --num_objects_training 7 \
    --seed 42 \
    --prediction_level super \
    --object_detection_model mask-rcnn \
    --wandb_project coco_crop_vs_circle \
    --wandb_run_name coco_crop_sum

