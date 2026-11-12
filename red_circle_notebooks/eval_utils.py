"""
synonyms and metrics for the concept quality test 
"""
import json

import h5py
import numpy as np
import splice

ROOT = "/workspaces/Object-Centric-Concept-Bottlenecks"
OBJ_FILE = "obj_7_{mode}_min_0.01_max_0.85_score_0.2_iou_0.5_model_mask-rcnn_0.25.h5"
PATHS = {
    ds: {
        "full":   f"{ROOT}/encodings/{ds}/clip/splice_embeddings_0.25.h5",
        "crop":   f"{ROOT}/encodings/{ds}/obj_encodings/clip/{OBJ_FILE.format(mode='crop')}",
        "circle": f"{ROOT}/encodings/{ds}/obj_encodings/clip/{OBJ_FILE.format(mode='circle')}",
    }
    for ds in ("voc", "coco")
}
METHODS     = ("crop", "circle")
NUM_OBJECTS = 7        
N_SUBSET    = 2000
SEED        = 0
TOP_KS      = (10, 20)
SEARCH_K    = 100

vocab  = splice.get_vocabulary("laion", vocabulary_size=10000)
vocab_idx = {w: i for i, w in enumerate(vocab)}

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]   
with open(f"{ROOT}/datasets/coco/annotations/instances_val2017.json") as f:
    _cats = json.load(f)["categories"]
COCO_CLASSES = [None] * 90            
for c in _cats:
    COCO_CLASSES[c["id"] - 1] = c["name"]
CLASS_NAMES = {"voc": VOC_CLASSES, "coco": COCO_CLASSES}


def draw_subset(ds):
    """ random images from the test split with >= 1 GT label, identical for crop and circle"""
    with h5py.File(PATHS[ds]["full"], "r") as f:
        labels = f["test_category"][:]
    eligible = np.flatnonzero(labels.sum(1) > 0)
    rows = np.sort(np.random.default_rng(SEED).choice(eligible, size=N_SUBSET, replace=False))
    return rows, labels[rows], len(labels), len(eligible)


def load_subset(ds, rows):
    """Full-image encoding + object encodings"""
    with h5py.File(PATHS[ds]["full"], "r") as f:
        img_ids = f["test_img_ids"][:].ravel()[rows]
        full    = f["test_embeddings"][rows]
    data = {"img_ids": img_ids, "full": full}
    for mode in METHODS:
        with h5py.File(PATHS[ds][mode], "r") as f:
            id2row  = {iid: r for r, iid in enumerate(f["test_obj_img_ids"][:].ravel())}
            num_obj = f["test_num_obj"][:].ravel()
            group   = f["test_obj_embeddings"]
            objs, n_obj = [], []
            for iid in img_ids:
                r = id2row[iid]
                n = min(int(num_obj[r]), NUM_OBJECTS)
                objs.append(group[f"obj_embeddings_{r}"][()][:n])   
                n_obj.append(n)
        data[mode], data[f"{mode}_num_obj"] = objs, np.array(n_obj)
    return data


# Synonym lists (exact word matching)
# each word belongs to at most one class per dataset

_PERSON = ["person", "persons", "people", "man", "men", "woman", "women", "human", "humans",
           "boy", "boys", "girl", "girls", "child", "children", "kid", "kids", "baby", "babies",
           "toddler", "toddlers", "adult", "adults", "teen", "teens", "teenager", "teenagers",
           "guy", "guys", "lady", "ladies", "gentleman"]
_AIRPLANE = ["airplane", "airplanes", "aircraft", "plane", "planes", "jet", "jets"]
_BICYCLE = ["bicycle", "bicycles", "bike", "bikes"]
_BIRD = ["bird", "birds", "sparrow", "eagle", "eagles", "robin", "parrot", "owl", "owls",
         "pigeon", "pigeons", "seagull", "gull", "duck", "ducks", "goose", "geese", "swan", "swans",
         "hawk", "pelican", "penguin", "hummingbird", "heron", "flamingo"]
_BOAT = ["boat", "boats", "ship", "ships", "vessel", "vessels", "yacht", "yachts", "canoe", "kayak", "ferry"]
_BOTTLE = ["bottle", "bottles", "flask"]
_BUS = ["bus", "buses"]
_CAR = ["car", "cars", "automobile", "sedan", "auto", "autos", "coupe", "hatchback", "suv", "taxi", "jeep"]
_CAT = ["cat", "cats", "kitten", "kittens", "kitty"]
_CHAIR = ["chair", "chairs", "armchair", "stool"]
_COW = ["cow", "cows", "cattle", "bull", "bulls", "ox", "calf"]
_TABLE = ["table", "tables"]
_DOG = ["dog", "dogs", "puppy", "puppies", "canine", "hound", "pup", "pups",
        "terrier", "retriever", "labrador", "poodle", "bulldog", "beagle"]
_HORSE = ["horse", "horses", "pony", "stallion", "mare", "foal"]
_MOTORBIKE = ["motorbike", "motorcycle", "motorcycles"]
_PLANT = ["plant", "plants", "potted", "planter", "planters", "succulent", "bonsai"]
_SHEEP = ["sheep", "lamb", "ram"]
_SOFA = ["sofa", "couch", "sectional"]
_TRAIN = ["train", "trains", "locomotive", "tram"]

VOC_SYNONYMS = {
    "aeroplane":   _AIRPLANE,
    "bicycle":     _BICYCLE,
    "bird":        _BIRD,
    "boat":        _BOAT,
    "bottle":      _BOTTLE,
    "bus":         _BUS,
    "car":         _CAR,
    "cat":         _CAT,
    "chair":       _CHAIR,
    "cow":         _COW,
    "diningtable": _TABLE,
    "dog":         _DOG,
    "horse":       _HORSE,
    "motorbike":   _MOTORBIKE,
    "person":      _PERSON,
    "pottedplant": _PLANT,
    "sheep":       _SHEEP,
    "sofa":        _SOFA,
    "train":       _TRAIN,
    "tvmonitor":   ["tv", "television", "monitor", "monitors", "screen", "screens"],
}

COCO_SYNONYMS = {
    "person":         _PERSON,
    "bicycle":        _BICYCLE,
    "car":            _CAR,
    "motorcycle":     _MOTORBIKE,
    "airplane":       _AIRPLANE,
    "bus":            _BUS,
    "train":          _TRAIN,
    "truck":          ["truck", "trucks", "lorry", "pickup"],
    "boat":           _BOAT,
    "traffic light":  [],
    "fire hydrant":   [],
    "stop sign":      ["stop"],
    "parking meter":  ["meter"],
    "bench":          ["bench", "benches"],
    "bird":           _BIRD,
    "cat":            _CAT,
    "dog":            _DOG,
    "horse":          _HORSE,
    "sheep":          _SHEEP,
    "cow":            _COW,
    "elephant":       ["elephant", "elephants"],
    "bear":           ["bear", "bears", "grizzly", "panda"],
    "zebra":          ["zebra"],
    "giraffe":        ["giraffe"],
    "backpack":       ["backpack", "backpacks"],
    "umbrella":       ["umbrella", "umbrellas"],
    "handbag":        ["handbag", "handbags", "purse", "tote"],
    "tie":            ["tie", "ties"],
    "suitcase":       ["suitcase", "luggage", "baggage"],
    "frisbee":        [],
    "skis":           ["skis", "ski"],
    "snowboard":      ["snowboard"],
    "sports ball":    ["ball", "balls", "football", "basketball", "volleyball"],
    "kite":           ["kite"],
    "baseball bat":   ["bat"],
    "baseball glove": ["glove", "gloves"],
    "skateboard":     ["skateboard"],
    "surfboard":      [],
    "tennis racket":  ["racket"],
    "bottle":         _BOTTLE,
    "wine glass":     ["glass"],
    "cup":            ["cup", "cups", "mug", "mugs"],
    "fork":           ["fork"],
    "knife":          ["knife", "knives"],
    "spoon":          ["spoon", "spoons"],
    "bowl":           ["bowl", "bowls"],
    "banana":         ["banana", "bananas"],
    "apple":          ["apple", "apples"],
    "sandwich":       ["sandwich", "sandwiches", "burger"],
    "orange":         ["orange", "oranges"],
    "broccoli":       ["broccoli"],
    "carrot":         ["carrot", "carrots"],
    "hot dog":        ["sausage"],
    "pizza":          ["pizza"],
    "donut":          ["donut", "donuts"],
    "cake":           ["cake", "cakes", "cupcake"],
    "chair":          _CHAIR,
    "couch":          _SOFA,
    "potted plant":   _PLANT,
    "bed":            ["bed", "beds", "mattress"],
    "dining table":   _TABLE,
    "toilet":         ["toilet", "wc"],
    "tv":             ["tv", "television", "monitor", "monitors"],
    "laptop":         ["laptop", "laptops", "macbook"],
    "mouse":          ["mouse"],
    "remote":         ["remote", "controller"],
    "keyboard":       ["keyboard"],
    "cell phone":     ["phone", "phones", "smartphone", "smartphones", "cellphone", "iphone"],
    "microwave":      ["microwave"],
    "oven":           ["oven", "stove"],
    "toaster":        [],
    "sink":           ["sink", "sinks"],
    "refrigerator":   ["refrigerator", "fridge", "freezer"],
    "book":           ["book", "books"],
    "clock":          ["clock", "clocks"],
    "vase":           ["vase"],
    "scissors":       ["scissors"],
    "teddy bear":     ["teddy"],
    "hair drier":     ["dryer"],
    "toothbrush":     ["toothbrush"],
}

SYNONYMS = {"voc": VOC_SYNONYMS, "coco": COCO_SYNONYMS}


def check_synonyms(syn, classes, name):
    """Check completeness, exact vocabulary coverage and uniqueness across classes"""
    assert set(syn) == {c for c in classes if c is not None}, f"{name}: classes do not match"
    missing = {c: [w for w in words if w not in vocab_idx] for c, words in syn.items()}
    assert not any(missing.values()), f"{name}: not in vocabulary: {missing}"
    owner = {}
    for c, words in syn.items():
        for w in words:
            assert w not in owner, f"{name}: '{w}' in '{owner[w]}' and '{c}'"
            owner[w] = c
    empty = [c for c, words in syn.items() if not words]
    print(f"{name}: {len(syn)} classes, {len(owner)} words, all exactly in vocabulary, unique "
          f"| not coverable: {empty or '–'}")


def aggregate(full, objs):
    """Sum and count over full image + objects"""
    stack = np.vstack([full[None], objs])
    return stack.sum(0), np.count_nonzero(stack, axis=0)


def rank_concepts(sum_vec, count_vec, how):
    """
    Sum:   sum value descending.
    Count: count descending, ties broken by sum value descending
    """
    active = np.flatnonzero(count_vec > 0)
    if how == "sum":
        order = np.argsort(-sum_vec[active], kind="stable")
    else:
        order = np.lexsort((-sum_vec[active], -count_vec[active]))
    return active[order]


def best_positions(ranking, gt_cols, label_sets):
    pos = {c: p + 1 for p, c in enumerate(ranking[:SEARCH_K])}
    best = []
    for col in gt_cols:
        hits = [pos[c] for c in label_sets[col] if c in pos]
        best.append(min(hits) if hits else None)
    return best


def evaluate(data, labels, ds):
    names, syn = CLASS_NAMES[ds], SYNONYMS[ds]
    label_sets = {col: {vocab_idx[w] for w in syn[n]} for col, n in enumerate(names) if n is not None}
    # labels without vocabulary entry cannot be covered 
    all_gt     = [list(np.flatnonzero(l)) for l in labels]
    gt_per_img = [[c for c in g if label_sets[c]] for g in all_gt]
    keep = [i for i, g in enumerate(gt_per_img) if g]
    removed = {
        "labels_removed": sum(len(a) - len(g) for a, g in zip(all_gt, gt_per_img)),
        "images_affected": sum(len(a) != len(g) for a, g in zip(all_gt, gt_per_img)),
        "images_excluded": len(labels) - len(keep),
        "pairs": sum(len(gt_per_img[i]) for i in keep),
    }

    rows = []
    for mode in METHODS:
        aggs = [aggregate(data["full"][i], data[mode][i]) for i in keep]
        n_active = np.mean([np.count_nonzero(cnt) for _, cnt in aggs])
        for how in ("sum", "count"):
            hit      = {k: [] for k in TOP_KS}
            rec_pair = {k: [] for k in TOP_KS}   
            rec_img  = {k: [] for k in TOP_KS}   
            ranks = []
            for (s, cnt), i in zip(aggs, keep):
                best = best_positions(rank_concepts(s, cnt, how), gt_per_img[i], label_sets)
                found = [b for b in best if b is not None]
                if found:
                    ranks.append(min(found))
                for k in TOP_KS:
                    hk = [b is not None and b <= k for b in best]
                    hit[k].append(any(hk))
                    rec_pair[k].extend(hk)
                    rec_img[k].append(np.mean(hk))
            rows.append({
                "Method": mode.capitalize(),
                "Aggregation": how.capitalize(),
                "N": len(keep),
                **{f"Hit@{k}": np.mean(hit[k]) for k in TOP_KS},
                "Found": len(ranks) / len(keep),
                "Found_n": len(ranks),
                "MeanRank": np.mean(ranks) if ranks else float("nan"),
                **{f"Recall@{k}": np.mean(rec_pair[k]) for k in TOP_KS},
                **{f"Recall@{k}_img": np.mean(rec_img[k]) for k in TOP_KS},
                "ActiveConcepts": n_active,
            })
    return rows, removed
