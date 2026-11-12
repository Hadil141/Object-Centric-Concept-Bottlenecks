"""
Builds a SPLICE-compatible dictionary from the 80 COCO category names
"""
import json
import torch
import splice


def load_category_names(instances_json_path):
    """
    Returns the 80 COCO category names, ordered by category id.
    """
    with open(instances_json_path) as f:
        data = json.load(f)
    categories = sorted(data["categories"], key=lambda c: c["id"])
    return [c["name"] for c in categories]


def build_coco_category_dictionary(splice_model, model_name, category_names, device):
    """
    Encodes each COCO category name via the CLIP text encoder and builds 
    a normalized, mean-centered dictionary matrix, same steps as 
    splice/splice.py::load(), just with the 80 COCO category names 
    instead of the LAION vocab list.
    """
    tokenizer = splice.get_tokenizer(model_name)
    concepts = []
    with torch.no_grad():
        for name in category_names:
            tokens = tokenizer(name).to(device)
            concept_embedding = splice_model.clip.encode_text(tokens)
            concepts.append(concept_embedding)

    dictionary = torch.nn.functional.normalize(torch.stack(concepts).squeeze(1), dim=1)
    dictionary = torch.nn.functional.normalize(dictionary - torch.mean(dictionary, dim=0), dim=1)
    return dictionary
