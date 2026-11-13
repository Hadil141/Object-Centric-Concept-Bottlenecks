"""
Typ 0 (Image-Level Intervention) COCO vocab
"""
import os

import torch

from utils import obj_enc_aggregation
from coco_category_vocab import load_category_names
from intervention_image_level import (
    load_intervention_testset,
    load_gt_categories,
    compute_percentile_targets,
    load_category_id_by_name,
    load_checkpoint,
    balanced_accuracy,
    load_variants,
    positive_variant_accuracy,
    load_boundary_type,
    nb_accuracy,
    apply_intervention,
)


def build_trivial_lexicon(instances_json_path):
    category_names = load_category_names(instances_json_path)
    return {name: [idx] for idx, name in enumerate(category_names)}


if __name__ == "__main__":
    lexicon = build_trivial_lexicon("datasets/coco/annotations/instances_val2017.json")
    category_id_by_name = load_category_id_by_name("data/cocologic/cocologic_train.json")

    base_embeddings_path = "encodings/cocologic/clip/splice_embeddings_0.25_coco_categories.h5"
    obj_embeddings_path = "encodings/cocologic/obj_encodings/clip/obj_7_crop_min_0.01_max_0.85_score_0.2_iou_0.5_model_mask-rcnn_0.25_coco_categories.h5"

    all_targets = {}
    for aggregation in ["sum", "max"]:
        targets_path = f"data/cocologic/percentile_targets_full_train_{aggregation}_coco_categories.pt"
        if os.path.exists(targets_path):
            all_targets[aggregation] = torch.load(targets_path)
            print(f"Loaded cached percentile targets from {targets_path}")
        else:
            targets = compute_percentile_targets(
                base_embeddings_path=base_embeddings_path,
                obj_embeddings_path=obj_embeddings_path,
                train_json_path="data/cocologic/cocologic_train.json",
                lexicon=lexicon,
                category_id_by_name=category_id_by_name,
                aggregation=aggregation,
                num_objects_training=7,
            )
            torch.save(targets, targets_path)
            print(f"Saved percentile targets for {len(targets)} categories ({aggregation}).")
            all_targets[aggregation] = targets

    embeddings, obj_embeddings, img_ids, labels = load_intervention_testset(
        base_embeddings_path, obj_embeddings_path, num_objects_training=7,
    )
    labels = labels.float()

    gt_categories = load_gt_categories("data/cocologic/cocologic_test.json", img_ids)
    variants = load_variants("data/cocologic/cocologic_test.json", img_ids)
    boundary_type = load_boundary_type("data/cocologic/cocologic_test.json", img_ids)

    for aggregation in ["sum", "max"]:
        model = load_checkpoint(f"checkpoints/cocologic/{aggregation}_super_multilabel_seed42_coco_categories.pt")
        obj_emb = obj_embeddings[:, :7, :]
        aggregated = obj_enc_aggregation(aggregation, obj_emb, embeddings, only_objects=False)

        with torch.no_grad():
            baseline_logits = model(aggregated)
        baseline_bacc = balanced_accuracy(baseline_logits, labels)
        baseline_pv = positive_variant_accuracy(baseline_logits, variants)
        baseline_nb = nb_accuracy(baseline_logits, boundary_type)

        intervened = apply_intervention(aggregated, gt_categories, all_targets[aggregation], category_id_by_name)
        with torch.no_grad():
            intervened_logits = model(intervened)
        intervened_bacc = balanced_accuracy(intervened_logits, labels)
        intervened_pv = positive_variant_accuracy(intervened_logits, variants)
        intervened_nb = nb_accuracy(intervened_logits, boundary_type)

        print(f"--- {aggregation} ---")
        print("baseline   B-Acc:", [round(v, 3) for v in baseline_bacc.tolist()], "mean:", round(torch.nanmean(baseline_bacc).item(), 4))
        print("intervened B-Acc:", [round(v, 3) for v in intervened_bacc.tolist()], "mean:", round(torch.nanmean(intervened_bacc).item(), 4))
        print("Δ B-Acc mean:", round((torch.nanmean(intervened_bacc) - torch.nanmean(baseline_bacc)).item(), 4))
        print("baseline   PosVar-Acc:", [round(v, 3) for v in baseline_pv.tolist()], "mean:", round(torch.nanmean(baseline_pv).item(), 4))
        print("intervened PosVar-Acc:", [round(v, 3) for v in intervened_pv.tolist()], "mean:", round(torch.nanmean(intervened_pv).item(), 4))
        print("Δ PosVar-Acc mean:", round((torch.nanmean(intervened_pv) - torch.nanmean(baseline_pv)).item(), 4))
        print("baseline   NB-Acc:", [round(v, 3) for v in baseline_nb.tolist()], "mean:", round(torch.nanmean(baseline_nb).item(), 4))
        print("intervened NB-Acc:", [round(v, 3) for v in intervened_nb.tolist()], "mean:", round(torch.nanmean(intervened_nb).item(), 4))
        print("Δ NB-Acc mean:", round((torch.nanmean(intervened_nb) - torch.nanmean(baseline_nb)).item(), 4))
