from functools import partial
import json
import os
from typing import Callable, Dict, List

import faiss
import numpy as np
import polars as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from src.datasets.mp16 import MP16Dataset, collate_fn
from src.pipeline.feature_extractor import FeatureExtractor
import torch


def build_index(d: int = 768):
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    return index


def add_record_to_index(index: faiss.IndexHNSWFlat, embeddings: np.ndarray):
    faiss.normalize_L2(embeddings)
    index.add(embeddings)


def save_index(index: faiss.IndexHNSWFlat, metadata: List[Dict], target_dir: str):
    faiss.write_index(index, os.path.join(target_dir, "index.index"))
    with open(os.path.join(target_dir, "metadata.json"), "w") as f:
        json.dump({"metadata": metadata}, f)


def read_index(target_dir: str):
    index = faiss.read_index(os.path.join(target_dir, "index.index"))
    with open(os.path.join(target_dir, "metadata.json"), "r") as f:
        metadata = json.loads(f.read())

    return index, metadata


def ingest(
    batch_encode_fn: Callable = None,
    ref_path: str = None,
    batch_collate_fn: Callable = None,
    batch_size: int = 128,
):
    if ref_path is None:
        ref_path = "./datasets/mp16-reason-train.csv"
    if batch_collate_fn is None:
        batch_collate_fn = collate_fn

    index = build_index(768)

    df = pl.read_csv(ref_path)
    dataset = MP16Dataset(df, img_col="IMG_ID", img_base_path="./datasets/mp16-reason")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=batch_collate_fn)

    if batch_encode_fn is None:
        batch_encode_fn = encode_fn

    all_outputs = batch_encode_fn(loader)

    metadatas = []
    for i, ref_item in tqdm(
        enumerate(df.to_dicts()), total=len(df), desc="ingest to index"
    ):
        for n in range(len(all_outputs["labels"][i])):
            metadatas.append(ref_item)

        _embed = all_outputs["alpha_embeddings"][i].astype("float32")
        add_record_to_index(index, _embed)
        break

    save_index(index, metadatas, target_dir="./checkpoints/")


def encode_fn(loader: DataLoader):
    extractor = FeatureExtractor(alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit20m_fultune_2xe.pth")
    all_outputs = {}
    for batch, target, images in tqdm(loader, desc="Mask"):
        out = extractor(batch, images, target)
        for key, val in out.items():
            if key not in all_outputs:
                all_outputs[key] = val
            else:
                all_outputs[key].extend(val)
        break

    return all_outputs


def collate_fn_clip(processor, batch):
    images = [b["image"] for b in batch]
    inputs = processor(images=images, return_tensors="pt")
    return inputs



def encode_clip(loader: DataLoader):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    clip_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model = clip_model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            out = clip_model.get_image_features(**{k: v.to(device) for k, v in batch.items()})
            out = out.pooler_output.cpu()
            embeddings.append(out)
            break

    embeddings = torch.vstack(embeddings).numpy()

    return embeddings


if __name__ == "__main__":
    # ingest(batch_size=16)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    ingest(batch_encode_fn=encode_clip, batch_collate_fn=partial(collate_fn_clip, clip_processor), batch_size=16)