python generate_concepts.py \
    --dataset voc \
    --data_dir data/voc \
    --encoding_dir encodings

python generate_concepts.py \
    --dataset voc \
    --data_dir data/voc \
    --use_object_concepts \
    --num_objects 7 \
    --min_object_size 0.01 \
    --max_object_size 0.85 \
    --min_score 0.2 \
    --max_iou_threshold 0.5 \
    --object_detection_model mask-rcnn \
    --object_encoding_dir obj_encodings



