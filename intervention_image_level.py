"""
Type 0 intervention (LAION)
"""
import json
import sys
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from train import load_splice_embeddings
from utils import obj_enc_aggregation
from model import LinearPredictor
import os


def build_laion_lexicon():
    """
    from curated synonym lists in red_circle_notebooks/eval_utils.py 
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "red_circle_notebooks"))
    from eval_utils import COCO_SYNONYMS, vocab_idx
    return {cat: [vocab_idx[w] for w in words] for cat, words in COCO_SYNONYMS.items()}


def load_intervention_testset(embeddings_path, obj_embeddings_path, num_objects_training, split="test"):
    with h5py.File(embeddings_path, "r") as f:
        embeddings = torch.from_numpy(f[f"{split}_embeddings"][:])
        labels = torch.from_numpy(f[f"{split}_supercategory"][:])  
        img_ids = torch.from_numpy(f[f"{split}_img_ids"][:])

    with h5py.File(obj_embeddings_path, "r") as f:
        obj_group = f[f"{split}_obj_embeddings"]
        obj_embeddings_list = [
            torch.from_numpy(obj_group[k][:])
            for k in sorted(obj_group.keys(), key=lambda x: int(x.split("_")[-1]))
        ]
        obj_embeddings = torch.stack([
            F.pad(t, (0, 0, 0, num_objects_training - t.shape[0]))
            for t in obj_embeddings_list
        ])
        obj_img_ids = torch.from_numpy(f[f"{split}_obj_img_ids"][:])

    # image ids must be unique and identical in both files
    base_ids, obj_ids = img_ids.flatten().tolist(), obj_img_ids.flatten().tolist()
    assert len(set(base_ids)) == len(base_ids) and len(set(obj_ids)) == len(obj_ids)
    assert set(base_ids) == set(obj_ids)

    # align object embeddings to the base embedding(image embedding) order via image_id,
    id_to_index = {iid.item(): idx for idx, iid in enumerate(obj_img_ids)}
    aligned = torch.tensor([id_to_index[iid.item()] for iid in img_ids], dtype=torch.long)
    obj_embeddings = obj_embeddings[aligned]

    return embeddings, obj_embeddings, img_ids, labels


def load_gt_categories(json_path, img_ids):

    with open(json_path) as f:
        data = json.load(f)
    images = data["images"]
    return torch.stack([
        torch.tensor(images[str(iid.item())]["categories"], dtype=torch.int64)
        for iid in img_ids
    ])

def compute_percentile_targets(
    base_embeddings_path, obj_embeddings_path, train_json_path,
    lexicon, category_id_by_name, aggregation, num_objects_training,
    full_train=True,):
    """
    Compute per concept boost/suppress targets on the aggregated vectors
    """
    if full_train:
        train_embeddings, train_obj_embeddings, train_img_ids, _ = load_intervention_testset(
            base_embeddings_path, obj_embeddings_path, num_objects_training, split="train",
        )
    else:
        train_dataset, _, _, _, _ = load_splice_embeddings(
            base_embeddings_path, obj_embeddings_path,
            num_objects_training=num_objects_training,
        )
        train_embeddings, _, train_obj_embeddings, train_img_ids, _ = train_dataset.tensors

    obj_embeddings = train_obj_embeddings[:, :num_objects_training, :]
    aggregated = obj_enc_aggregation(aggregation, obj_embeddings, train_embeddings, only_objects=False)

    gt_categories = load_gt_categories(train_json_path, train_img_ids)

    targets = {}
    for cat, idxs in lexicon.items():
        if not idxs or cat not in category_id_by_name:
            print(f"No concept group for '{cat}', skipped.")
            continue
        present = gt_categories[:, category_id_by_name[cat]] > 0
        if present.sum() == 0 or (~present).sum() == 0:
            continue  
        boost, suppress = {}, {}
        for i in idxs:
            values = aggregated[:, i].numpy()
            boost[i] = float(np.percentile(values[present.numpy()], 95))
            suppress[i] = float(np.percentile(values[~present.numpy()], 5))
        targets[cat] = {"boost": boost, "suppress": suppress}
    return targets


def load_category_id_by_name(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {c["name"]: c["id"] for c in data["categories"]}


def load_checkpoint(checkpoint_path, device="cpu"):
    """
    Reconstruct the frozen LinearPredictor 
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LinearPredictor(
        num_classes=checkpoint["num_classes"],
        explicit_encoding_dim=checkpoint["encoding_dim"],
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model

def balanced_accuracy(logits, labels, threshold=0.5):

    probabilities = torch.sigmoid(logits)
    predicted = (probabilities > threshold).int()
    labels = labels.int()

    num_rules = labels.shape[1]
    b_acc = torch.zeros(num_rules)
    for r in range(num_rules):
        pos, neg = labels[:, r] == 1, labels[:, r] == 0
        sensitivity = (predicted[pos, r] == 1).float().mean() if pos.any() else float("nan")
        specificity = (predicted[neg, r] == 0).float().mean() if neg.any() else float("nan")
        b_acc[r] = (sensitivity + specificity) / 2
    return b_acc

def load_variants(json_path, img_ids):

    with open(json_path) as f:
        data = json.load(f)
    images = data["images"]

    max_variant_id = max(
        (vid for img in images.values() for rv in img["rule_variants"] for vid in rv),
        default=0,
    )
    variant_tensors = []
    for iid in img_ids:
        rule_variants_list = images[str(iid.item())]["rule_variants"]
        v = torch.zeros((len(rule_variants_list), max_variant_id + 1), dtype=torch.float32)
        for i, rule_variants in enumerate(rule_variants_list):
            for variant_id in rule_variants:
                v[i, variant_id] = 1.0
        variant_tensors.append(v)
    return torch.stack(variant_tensors)

def positive_variant_accuracy(logits, variants, threshold=0.5):

    probabilities = torch.sigmoid(logits)
    predicted = (probabilities > threshold).int()

    num_rules, num_variants = variants.shape[1], variants.shape[2]
    pv_acc = torch.zeros(num_rules)
    for r in range(num_rules):
        variant_accs = [
            (predicted[variants[:, r, v] > 0, r] == 1).float().mean()
            for v in range(num_variants)
            if (variants[:, r, v] > 0).any()
        ]
        pv_acc[r] = torch.stack(variant_accs).mean() if variant_accs else float("nan")
    return pv_acc

def load_boundary_type(json_path, img_ids):

    with open(json_path) as f:
        data = json.load(f)
    images = data["images"]
    return torch.stack([
        torch.tensor(images[str(iid.item())]["boundary_type"], dtype=torch.int64)
        for iid in img_ids
    ])

def nb_accuracy(logits, boundary_type, threshold=0.5):

    probabilities = torch.sigmoid(logits)
    predicted = (probabilities > threshold).int()

    num_rules = boundary_type.shape[1]
    nb_acc = torch.zeros(num_rules)
    for r in range(num_rules):
        nb_values = [v for v in boundary_type[:, r].unique().tolist() if v != 0]
        group_accs = [
            (predicted[boundary_type[:, r] == v, r] == 0).float().mean()
            for v in nb_values
        ]
        nb_acc[r] = torch.stack(group_accs).mean() if group_accs else float("nan")
    return nb_acc

def apply_intervention(aggregated, gt_categories, targets, category_id_by_name, suppress=True):
    """
    absent categories -> suppress value
    present categories -> boost value
    """
    intervened = aggregated.clone()
    if suppress:
        for cat, t in targets.items():
            absent = gt_categories[:, category_id_by_name[cat]] == 0
            for idx, suppress_val in t["suppress"].items():
                intervened[absent, idx] = suppress_val
    for cat, t in targets.items():
        present = gt_categories[:, category_id_by_name[cat]] > 0
        for idx, boost_val in t["boost"].items():
            intervened[present, idx] = boost_val
    return intervened

def stepwise_intervention_curve(model, aggregated, gt_categories, targets, category_id_by_name, labels,
                                 variants, boundary_type, orders, suppress=True):

    curves = {"bacc": [], "pv": [], "nb": []}
    for category_order in orders:
        baccs, pvs, nbs = [], [], []
        for n in range(len(category_order) + 1):
            targets_n = {cat: targets[cat] for cat in category_order[:n] if cat in targets}
            intervened_n = apply_intervention(aggregated, gt_categories, targets_n, category_id_by_name,
                                              suppress=suppress)
            with torch.no_grad():
                logits_n = model(intervened_n)
            baccs.append(torch.nanmean(balanced_accuracy(logits_n, labels)).item())
            pvs.append(torch.nanmean(positive_variant_accuracy(logits_n, variants)).item())
            nbs.append(torch.nanmean(nb_accuracy(logits_n, boundary_type)).item())
        curves["bacc"].append(baccs)
        curves["pv"].append(pvs)
        curves["nb"].append(nbs)

    ns = list(range(len(orders[0]) + 1))
    curves = {k: np.array(v) for k, v in curves.items()}
    stats = {k: (v.mean(axis=0), v.std(axis=0)) for k, v in curves.items()}
    return ns, stats, curves


def compute_coverage(gt_categories, targets, category_id_by_name):

    touched = torch.zeros(gt_categories.shape[0], dtype=torch.bool)
    for cat in targets:
        present = gt_categories[:, category_id_by_name[cat]] > 0
        touched |= present
    return touched

def flip_analysis(baseline_logits, intervened_logits, labels, threshold=0.5):

    baseline_pred = (torch.sigmoid(baseline_logits) > threshold).int()
    intervened_pred = (torch.sigmoid(intervened_logits) > threshold).int()
    labels = labels.int()

    baseline_correct = baseline_pred == labels
    intervened_correct = intervened_pred == labels

    results = []
    for r in range(labels.shape[1]):
        results.append({
            "fixed":       int((~baseline_correct[:, r] & intervened_correct[:, r]).sum()),
            "regressed":   int((baseline_correct[:, r] & ~intervened_correct[:, r]).sum()),
            "still_wrong": int((~baseline_correct[:, r] & ~intervened_correct[:, r]).sum()),
            "unaffected":  int((baseline_correct[:, r] & intervened_correct[:, r]).sum()),
        })
    return results


if __name__ == "__main__":
    lexicon = build_laion_lexicon()
    category_id_by_name = load_category_id_by_name("data/cocologic/cocologic_train.json")

    base_embeddings_path = "encodings/cocologic/clip/splice_embeddings_0.25.h5"
    obj_embeddings_path = "encodings/cocologic/obj_encodings/clip/obj_7_crop_min_0.01_max_0.85_score_0.2_iou_0.5_model_mask-rcnn_0.25.h5"

    # Perzentil-Ziele (gecacht, wird nur einmal berechnet)
    all_targets = {}
    for aggregation in ["sum", "max"]:
        targets_path = f"data/cocologic/percentile_targets_full_train_{aggregation}_laion.pt"
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

    # Baseline, Intervention, Vergleich
    embeddings, obj_embeddings, img_ids, labels = load_intervention_testset(
        base_embeddings_path, obj_embeddings_path, num_objects_training=7,
    )
    labels = labels.float()

    gt_categories = load_gt_categories("data/cocologic/cocologic_test.json", img_ids)
    variants = load_variants("data/cocologic/cocologic_test.json", img_ids)
    boundary_type = load_boundary_type("data/cocologic/cocologic_test.json", img_ids)


    for aggregation in ["sum", "max"]:
        model = load_checkpoint(f"checkpoints/cocologic/{aggregation}_super_multilabel_seed42.pt")
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
