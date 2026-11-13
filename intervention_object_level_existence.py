"""
Type 1 (object-level existence) intervention (LAION)
"""
import json
import numpy as np
import torch
import os
import h5py
from utils import obj_enc_aggregation


from scipy.optimize import linear_sum_assignment
from utils import calculate_iou
from intervention_image_level import (
    load_checkpoint,
    load_intervention_testset,
    load_gt_categories,
    load_variants,
    load_boundary_type,
    balanced_accuracy,
    positive_variant_accuracy,
    nb_accuracy,
)

def xywh_to_xyxy(box):
    """
    Convert a COCO bounding box from (x, y, w, h) format to (x1, y1, x2, y2) format
    """
    x, y, w, h = box
    return [x, y, x+w, y+h]

def rescale_box(box, from_size, to_size):

    from_w, from_h = from_size
    to_w, to_h = to_size
    scale_x = to_w / from_w
    scale_y = to_h / from_h
    x1, y1, x2, y2 = box
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

def load_original_image_sizes(instances_json_path):

    with open(instances_json_path) as f:
        data = json.load(f)
    return {img["id"]: (img["width"], img["height"]) for img in data["images"]}


def load_coco_gt_boxes(instances_json_path, iscrowd_exclude=True):

    with open(instances_json_path) as f:
        data = json.load(f)

    gt_boxes = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        if iscrowd_exclude and ann["iscrowd"] == 1:
            continue
        gt_boxes[ann["image_id"]].append({
            "bbox_xyxy": xywh_to_xyxy(ann["bbox"]),
            "category_id": ann["category_id"],
        })
    return gt_boxes

def _iou_matrix(pred_boxes, gt_boxes):
    return np.array([[calculate_iou(p, g) for g in gt_boxes] for p in pred_boxes])

def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5, strategy="greedy"):
    """
    Match predicted (Mask-R-CNN) boxes to GT boxes via IoU
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    ious = _iou_matrix(pred_boxes, gt_boxes)
    matched_pairs, used_pred, used_gt = [], set(), set()

    if strategy == "greedy":
        candidates = sorted(
            (
                (ious[p, g], p, g)
                for p in range(len(pred_boxes))
                for g in range(len(gt_boxes))
                if ious[p, g] >= iou_threshold
            ),
            reverse=True,
        )
        for iou, p, g in candidates:
            if p in used_pred or g in used_gt:
                continue
            matched_pairs.append((p, g, iou))
            used_pred.add(p)
            used_gt.add(g)

    elif strategy == "hungarian":
        pred_idxs, gt_idxs = linear_sum_assignment(1 - ious)
        for p, g in zip(pred_idxs, gt_idxs):
            if ious[p, g] >= iou_threshold:
                matched_pairs.append((p, g, ious[p, g]))
                used_pred.add(p)
                used_gt.add(g)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    unmatched_pred = [p for p in range(len(pred_boxes)) if p not in used_pred]
    unmatched_gt = [g for g in range(len(gt_boxes)) if g not in used_gt]
    return matched_pairs, unmatched_pred, unmatched_gt

def remove_unmatched(obj_embeddings, obj_boxes, unmatched_pred_idxs):
    """
    Drop object slots whose Mask-R-CNN box did not match any GT box

    """
    keep_mask = torch.ones(obj_embeddings.shape[0], dtype=torch.bool)
    keep_mask[unmatched_pred_idxs] = False
    return obj_embeddings[keep_mask], obj_boxes[keep_mask]

def encode_new_objects(image_224, gt_boxes_224, splice_model, preprocessor, device):
    """
    Crop each unmatched GT box from image_224 and encode with SpLiCE 
    """
    crops = []
    for box in gt_boxes_224:
        x1, y1, x2, y2 = [int(c) for c in box]
        crop = image_224[:, y1:y2, x1:x2]
        processed = preprocessor(crop.unsqueeze(0)).squeeze(0)
        crops.append(processed)

    batch = torch.stack(crops).to(device)
    embeddings, _ = splice_model.encode_image(batch)
    return embeddings

def apply_object_level_intervention(
    obj_embeddings, obj_boxes, image_224, gt_boxes_orig, orig_size,
    splice_model, preprocessor, device,
    iou_threshold=0.5, strategy="greedy", num_objects_training=7,
    relevant_category_ids=None,
):

    obj_embeddings = obj_embeddings[:obj_boxes.shape[0]]

    pred_boxes_224 = obj_boxes.tolist()
    gt_boxes_224 = [rescale_box(b["bbox_xyxy"], orig_size, (224, 224)) for b in gt_boxes_orig]
    gt_category_ids = [b["category_id"] for b in gt_boxes_orig]

    matched_pairs, unmatched_pred, unmatched_gt = match_boxes(
        pred_boxes_224, gt_boxes_224, iou_threshold=iou_threshold, strategy=strategy
    )
    pred_to_gt_category = {p: gt_category_ids[g] for p, g, _ in matched_pairs}

    kept_embeddings, kept_boxes = remove_unmatched(obj_embeddings, obj_boxes, unmatched_pred)
    kept_embeddings, kept_boxes = kept_embeddings.to(device), kept_boxes.to(device)
    after_remove_embeddings = kept_embeddings.clone()

    unmatched_pred_set = set(unmatched_pred)
    kept_category_ids = [
        pred_to_gt_category[p] for p in range(len(pred_boxes_224)) if p not in unmatched_pred_set
    ]

    num_removed = len(unmatched_pred)
    num_added = 0
    added_category_ids = []

    free_slots = num_objects_training - kept_embeddings.shape[0]
    if free_slots > 0 and len(unmatched_gt) > 0:
        unmatched_entries = [(gt_boxes_224[g], gt_category_ids[g]) for g in unmatched_gt]

        def sort_key(entry):
            box, cat_id = entry
            is_relevant = relevant_category_ids is not None and cat_id in relevant_category_ids
            area = (box[2] - box[0]) * (box[3] - box[1])
            return (is_relevant, area)

        unmatched_entries.sort(key=sort_key, reverse=True)
        entries_to_add = unmatched_entries[:free_slots]
        boxes_to_add = [e[0] for e in entries_to_add]
        added_category_ids = [e[1] for e in entries_to_add]
        num_added = len(boxes_to_add)

        new_embeddings = encode_new_objects(image_224, boxes_to_add, splice_model, preprocessor, device)
        new_boxes = torch.tensor(boxes_to_add, dtype=kept_boxes.dtype, device=device)

        kept_embeddings = torch.cat([kept_embeddings, new_embeddings], dim=0)
        kept_boxes = torch.cat([kept_boxes, new_boxes], dim=0)

    return kept_embeddings, kept_boxes, num_removed, num_added, added_category_ids, kept_category_ids, after_remove_embeddings

def load_test_objects_raw(obj_embeddings_path):
    with h5py.File(obj_embeddings_path, "r") as f:
        emb_group = f["test_obj_embeddings"]
        box_group = f["test_obj_boxes"]
        emb_keys = sorted(emb_group.keys(), key=lambda x: int(x.split("_")[-1]))
        box_keys = sorted(box_group.keys(), key=lambda x: int(x.split("_")[-1]))
        embeddings_list = [torch.from_numpy(emb_group[k][:]) for k in emb_keys]
        boxes_list = [torch.from_numpy(box_group[k][:]) for k in box_keys]
        obj_img_ids = torch.from_numpy(f["test_obj_img_ids"][:])
    return embeddings_list, boxes_list, obj_img_ids



if __name__ == "__main__":
    import splice
    import torchvision.transforms.v2 as T
    from PIL import Image
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"

    splice_model = splice.load(
        "open_clip:ViT-B-32", "laion", 10000, l1_penalty=0.25, device=device, return_weights=True
    )
    preprocessor = T.Compose([
        T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        T.CenterCrop(size=(224, 224)),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    base_embeddings_path = "encodings/cocologic/clip/splice_embeddings_0.25.h5"
    obj_embeddings_path = "encodings/cocologic/obj_encodings/clip/obj_7_crop_min_0.01_max_0.85_score_0.2_iou_0.5_model_mask-rcnn_0.25.h5"
    test_json_path = "data/cocologic/cocologic_test.json"
    instances_json_path = "datasets/coco/annotations/instances_val2017.json"
    image_base_path = "datasets/coco/images"
    aggregation = "sum"
    num_objects_training = 7
    iou_threshold = 0.5
    strategy = "greedy"
    subset_size = 3500

    model = load_checkpoint(f"checkpoints/cocologic/{aggregation}_super_multilabel_seed42.pt", device=device)

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

 