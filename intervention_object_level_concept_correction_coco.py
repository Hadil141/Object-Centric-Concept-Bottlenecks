"""
Typ 2 (Object-Level Concept Correction) coco vocab
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

from utils import obj_enc_aggregation
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
    rescale_box,
    load_original_image_sizes,
    load_coco_gt_boxes,
    match_boxes,
)
from coco_category_vocab import load_category_names


def load_objects_raw(obj_embeddings_path, split):

    with h5py.File(obj_embeddings_path, "r") as f:
        emb_group = f[f"{split}_obj_embeddings"]
        box_group = f[f"{split}_obj_boxes"]
        emb_keys = sorted(emb_group.keys(), key=lambda x: int(x.split("_")[-1]))
        box_keys = sorted(box_group.keys(), key=lambda x: int(x.split("_")[-1]))
        embeddings_list = [torch.from_numpy(emb_group[k][:]) for k in emb_keys]
        boxes_list = [torch.from_numpy(box_group[k][:]) for k in box_keys]
        obj_img_ids = torch.from_numpy(f[f"{split}_obj_img_ids"][:])
    return embeddings_list, boxes_list, obj_img_ids


def match_image_objects(obj_boxes, gt_boxes_orig, orig_size, iou_threshold, strategy):
    """
    Matches one images predicted object boxes to its GT boxes
    """
    pred_boxes_224 = obj_boxes.tolist()
    gt_boxes_224 = [rescale_box(b["bbox_xyxy"], orig_size, (224, 224)) for b in gt_boxes_orig]
    gt_category_ids = [b["category_id"] for b in gt_boxes_orig]

    matched_pairs, unmatched_pred, _ = match_boxes(
        pred_boxes_224, gt_boxes_224, iou_threshold=iou_threshold, strategy=strategy
    )
    return [(p, gt_category_ids[g]) for p, g, _ in matched_pairs], unmatched_pred


def compute_object_level_percentile_targets(
    train_obj_embeddings_path, train_instances_json_path, dim_by_category_id,
    iou_threshold=0.5, strategy="greedy",
):
    """
    boost/suppress values for concept dimensions, calculated over individual matched objects
    """
    embeddings_list, boxes_list, obj_img_ids = load_objects_raw(train_obj_embeddings_path, "train")
    gt_boxes_by_image = load_coco_gt_boxes(train_instances_json_path)
    orig_sizes = load_original_image_sizes(train_instances_json_path)

    num_dims = len(dim_by_category_id)
    present_values = [[] for _ in range(num_dims)]
    absent_values = [[] for _ in range(num_dims)]

    n_matched_total = 0
    for obj_emb, obj_boxes, image_id_t in zip(embeddings_list, boxes_list, obj_img_ids):
        image_id = image_id_t.item()
        if obj_boxes.shape[0] == 0:
            continue
        gt_boxes_orig = gt_boxes_by_image.get(image_id, [])
        if not gt_boxes_orig:
            continue
        orig_size = orig_sizes[image_id]

        matched_pairs, _ = match_image_objects(obj_boxes, gt_boxes_orig, orig_size, iou_threshold, strategy)
        for pred_idx, gt_cat_id in matched_pairs:
            if gt_cat_id not in dim_by_category_id:
                continue
            own_dim = dim_by_category_id[gt_cat_id]
            values = obj_emb[pred_idx].numpy()
            present_values[own_dim].append(values[own_dim])
            for dim in range(num_dims):
                if dim != own_dim:
                    absent_values[dim].append(values[dim])
            n_matched_total += 1

    boost, suppress = {}, {}
    for dim in range(num_dims):
        boost[dim] = float(np.percentile(present_values[dim], 95)) if present_values[dim] else 0.0
        suppress[dim] = float(np.percentile(absent_values[dim], 5)) if absent_values[dim] else 0.0

    print(f"Insgesamt {n_matched_total} gematchte Trainingsobjekte verwendet.")
    return {"boost": boost, "suppress": suppress}


def apply_object_level_concept_correction(
    obj_embeddings, obj_boxes, gt_boxes_orig, orig_size, targets, dim_by_category_id,
    iou_threshold=0.5, strategy="greedy",
):

    corrected = obj_embeddings.clone()
    if obj_boxes.shape[0] == 0 or not gt_boxes_orig:
        return corrected, 0

    matched_pairs, _ = match_image_objects(obj_boxes, gt_boxes_orig, orig_size, iou_threshold, strategy)
    num_dims = obj_embeddings.shape[1]
    boost, suppress = targets["boost"], targets["suppress"]

    for pred_idx, gt_cat_id in matched_pairs:
        if gt_cat_id not in dim_by_category_id:
            continue
        own_dim = dim_by_category_id[gt_cat_id]
        for dim in range(num_dims):
            corrected[pred_idx, dim] = boost[dim] if dim == own_dim else suppress[dim]

    return corrected, len(matched_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--aggregation", default="sum", choices=["sum", "concat"])
    args = parser.parse_args()

    device = "cpu"  

    train_instances_json_path = "datasets/coco/annotations/instances_train2017.json"
    test_instances_json_path = "datasets/coco/annotations/instances_val2017.json"
    category_names = load_category_names(test_instances_json_path)  
    category_id_by_name = load_category_id_by_name(test_instances_json_path)
    dim_by_category_id = {category_id_by_name[name]: i for i, name in enumerate(category_names)}

    base_embeddings_path = "encodings/cocologic/clip/splice_embeddings_0.25_coco_categories.h5"
    obj_embeddings_path = "encodings/cocologic/obj_encodings/clip/obj_7_crop_min_0.01_max_0.85_score_0.2_iou_0.5_model_mask-rcnn_0.25_coco_categories.h5"
    test_json_path = "data/cocologic/cocologic_test.json"
    aggregation = args.aggregation
    num_objects_training = 7
    iou_threshold = 0.5
    strategy = "greedy"

    targets_path = "data/cocologic/object_level_percentile_targets_coco_categories.pt"
    if os.path.exists(targets_path):
        targets = torch.load(targets_path)
        print(f"Loaded cached object-level percentile targets from {targets_path}")
    else:
        targets = compute_object_level_percentile_targets(
            obj_embeddings_path, train_instances_json_path, dim_by_category_id,
            iou_threshold=iou_threshold, strategy=strategy,
        )
        torch.save(targets, targets_path)
        print(f"Saved object-level percentile targets to {targets_path}")

    model = load_checkpoint(f"checkpoints/cocologic/{aggregation}_super_multilabel_seed42_coco_categories.pt", device=device)

    embeddings, obj_embeddings_padded, img_ids, labels = load_intervention_testset(
        base_embeddings_path, obj_embeddings_path, num_objects_training
    )
    labels = labels.float()
    variants = load_variants(test_json_path, img_ids)
    boundary_type = load_boundary_type(test_json_path, img_ids)

    baseline_aggregated = obj_enc_aggregation(
        aggregation, obj_embeddings_padded, embeddings, only_objects=False
    )
    with torch.no_grad():
        baseline_logits = model(baseline_aggregated).cpu()

    raw_embeddings_list, raw_boxes_list, obj_img_ids = load_objects_raw(obj_embeddings_path, "test")
    id_to_raw_idx = {iid.item(): idx for idx, iid in enumerate(obj_img_ids)}

    gt_boxes_by_image = load_coco_gt_boxes(test_instances_json_path)
    orig_sizes = load_original_image_sizes(test_instances_json_path)

    intervened_obj_list = []
    n_images_with_match = 0
    for i in range(len(img_ids)):
        image_id = img_ids[i].item()
        raw_idx = id_to_raw_idx[image_id]
        obj_emb = raw_embeddings_list[raw_idx]
        obj_boxes = raw_boxes_list[raw_idx]

        corrected, n_matched = apply_object_level_concept_correction(
            obj_emb, obj_boxes, gt_boxes_by_image.get(image_id, []), orig_sizes[image_id],
            targets, dim_by_category_id, iou_threshold=iou_threshold, strategy=strategy,
        )
        if n_matched > 0:
            n_images_with_match += 1

        padded = torch.zeros(num_objects_training, corrected.shape[1])
        n = min(corrected.shape[0], num_objects_training)
        padded[:n] = corrected[:n]
        intervened_obj_list.append(padded)

    intervened_obj = torch.stack(intervened_obj_list)
    intervened_aggregated = obj_enc_aggregation(aggregation, intervened_obj, embeddings, only_objects=False)
    with torch.no_grad():
        intervened_logits = model(intervened_aggregated).cpu()

    print(f"\nCoverage: {n_images_with_match}/{len(img_ids)} Testbilder ({n_images_with_match/len(img_ids):.1%}) hatten mindestens ein gematchtes Objekt.")

    RULE_NAMES = ["Signal and Ride", "Double Serving", "Herd Alone", "Dog or Car", "Three of a Kind",
                  "Car Majority", "Empty Seat", "Single Mode Traffic", "Personal Transport", "Surf Trip"]

    baseline_bacc = balanced_accuracy(baseline_logits, labels)
    intervened_bacc = balanced_accuracy(intervened_logits, labels)
    baseline_pv = positive_variant_accuracy(baseline_logits, variants)
    intervened_pv = positive_variant_accuracy(intervened_logits, variants)
    baseline_nb = nb_accuracy(baseline_logits, boundary_type)
    intervened_nb = nb_accuracy(intervened_logits, boundary_type)

    print(f"\n{'Regel':22s} | {'B-Acc':>18s} | {'PosVar':>18s} | {'NB-Acc':>18s}")
    for r in range(10):
        print(f"{RULE_NAMES[r]:22s} | {baseline_bacc[r]:.1%}->{intervened_bacc[r]:.1%} ({(intervened_bacc[r]-baseline_bacc[r])*100:+.1f}pp) | "
              f"{baseline_pv[r]:.1%}->{intervened_pv[r]:.1%} ({(intervened_pv[r]-baseline_pv[r])*100:+.1f}pp) | "
              f"{baseline_nb[r]:.1%}->{intervened_nb[r]:.1%} ({(intervened_nb[r]-baseline_nb[r])*100:+.1f}pp)")

    print(f"\nMittel B-Acc:  {torch.nanmean(baseline_bacc).item():.2%} -> {torch.nanmean(intervened_bacc).item():.2%}")
    print(f"Mittel PosVar: {torch.nanmean(baseline_pv).item():.2%} -> {torch.nanmean(intervened_pv).item():.2%}")
    print(f"Mittel NB-Acc: {torch.nanmean(baseline_nb).item():.2%} -> {torch.nanmean(intervened_nb).item():.2%}")
