"""
Typ 1 (Object-Level Existence Intervention) coco vocab
"""
import json
import os

import torch
import torch.nn.functional as F
import splice
import torchvision.transforms.v2 as T
from PIL import Image

from utils import obj_enc_aggregation
from coco_category_vocab import load_category_names, build_coco_category_dictionary
from intervention_image_level import (
    load_checkpoint,
    load_intervention_testset,
    load_variants,
    load_boundary_type,
    balanced_accuracy,
    positive_variant_accuracy,
    nb_accuracy,
    load_category_id_by_name,
)

from intervention_object_level_existence import (
    load_coco_gt_boxes,
    load_original_image_sizes,
    load_test_objects_raw,
    apply_object_level_intervention,
)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "open_clip:ViT-B-32"


    splice_model = splice.load(
        model_name, "laion", 10000, l1_penalty=0.25, device=device, return_weights=True
    )
    instances_json_path = "datasets/coco/annotations/instances_val2017.json"
    category_names = load_category_names(instances_json_path)
    splice_model.dictionary = build_coco_category_dictionary(
        splice_model, model_name, category_names, device
    ).to(device)

    RELEVANT_CATEGORIES = {
    "traffic light", "bicycle", "bus", "train", "bottle", "cup", "pizza",
    "cow", "elephant", "sheep", "dog", "car", "bowl", "truck",
    "couch", "chair", "motorcycle", "surfboard", "person",
    }
    category_id_by_name = load_category_id_by_name(instances_json_path)
    RELEVANT_CATEGORY_IDS = {category_id_by_name[name] for name in RELEVANT_CATEGORIES}


    preprocessor = T.Compose([
        T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        T.CenterCrop(size=(224, 224)),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    base_embeddings_path = "encodings/cocologic/clip/splice_embeddings_0.25_coco_categories.h5"
    obj_embeddings_path = "encodings/cocologic/obj_encodings/clip/obj_7_crop_min_0.01_max_0.85_score_0.2_iou_0.5_model_mask-rcnn_0.25_coco_categories.h5"
    test_json_path = "data/cocologic/cocologic_test.json"
    image_base_path = "datasets/coco/images"
    aggregation = "sum"
    num_objects_training = 7
    iou_threshold = 0.5
    strategy = "greedy"
    subset_size = 3500

    model = load_checkpoint(f"checkpoints/cocologic/{aggregation}_super_multilabel_seed42_coco_categories.pt", device=device)

    embeddings, obj_embeddings_padded, img_ids, labels = load_intervention_testset(
        base_embeddings_path, obj_embeddings_path, num_objects_training
    )
    labels = labels.float()
    variants = load_variants(test_json_path, img_ids)
    boundary_type = load_boundary_type(test_json_path, img_ids)

    raw_embeddings_list, raw_boxes_list, obj_img_ids = load_test_objects_raw(obj_embeddings_path)
    id_to_raw_idx = {iid.item(): idx for idx, iid in enumerate(obj_img_ids)}

    gt_boxes_by_image = load_coco_gt_boxes(instances_json_path)
    orig_sizes = load_original_image_sizes(instances_json_path)
    with open(test_json_path) as f:
        meta = json.load(f)

    subset = range(min(subset_size, len(img_ids)))

    baseline_aggregated = obj_enc_aggregation(
        aggregation, obj_embeddings_padded[list(subset)], embeddings[list(subset)], only_objects=False
    )
    with torch.no_grad():
        baseline_logits = model(baseline_aggregated.to(device)).cpu()

    intervened_obj_list = []
    for i in subset:
        image_id = img_ids[i].item()
        raw_idx = id_to_raw_idx[image_id]
        obj_emb = raw_embeddings_list[raw_idx].to(device)
        obj_boxes = raw_boxes_list[raw_idx].to(device)

        file_name = meta["images"][str(image_id)]["file_name"]
        image_224 = T.Compose([T.Resize((224, 224)), T.ToTensor()])(
            Image.open(os.path.join(image_base_path, file_name)).convert("RGB")
        ).to(device)

        new_emb, _, num_removed, num_added, _, _, _ = apply_object_level_intervention(
            obj_emb, obj_boxes, image_224,
            gt_boxes_by_image[image_id], orig_sizes[image_id],
            splice_model, preprocessor, device,
            iou_threshold=iou_threshold, strategy=strategy,
            num_objects_training=num_objects_training,
            relevant_category_ids=RELEVANT_CATEGORY_IDS,
        )
        padded = F.pad(new_emb, (0, 0, 0, num_objects_training - new_emb.shape[0]))
        intervened_obj_list.append(padded.cpu())

    intervened_obj = torch.stack(intervened_obj_list)
    intervened_aggregated = obj_enc_aggregation(
        aggregation, intervened_obj, embeddings[list(subset)], only_objects=False
    )
    with torch.no_grad():
        intervened_logits = model(intervened_aggregated.to(device)).cpu()

    idxs = list(subset)
    for name, logits in [("baseline", baseline_logits), ("intervened", intervened_logits)]:
        print(name)
        print(f"  B-Acc:  {balanced_accuracy(logits, labels[idxs])}")
        print(f"  PosVar: {positive_variant_accuracy(logits, variants[idxs])}")
        print(f"  NB-Acc: {nb_accuracy(logits, boundary_type[idxs])}")
        print()
