import argparse
import json
from functools import partial

import faiss
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from src import constant
from src.datasets.mp16 import MP16Dataset
from src.metrics import map_k, precision_k
from src.utils import clip_collate_fn, get_device, read_index, set_seed


def main(args):
    # prepare
    set_seed(args.seed)
    device = get_device()
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model = clip_model.eval()

    # dataset
    df_test = pl.read_csv(args.data_path)
    dataset = MP16Dataset(
        df_test,
        img_col="IMG_ID",
        img_base_path=args.img_base_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(clip_collate_fn, clip_processor),
    )
    index, ref_meta = read_index(args.index_dir)
    index.hnsw.efSearch = 128
    ref_meta = ref_meta["metadata"]

    # encode
    embeddings = []
    # pred_gps = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = clip_model.get_image_features(**batch)
            out = out.pooler_output.cpu()
            embeddings.append(out)

    embeddings = torch.vstack(embeddings).numpy()

    # similarity search
    all_ref_gps = []
    all_metas = {}
    for i in tqdm(range(len(df_test)), desc="eval"):
        query_emb = embeddings[i].astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_emb)
        sim, ind = index.search(query_emb, args.topk)
        flat_D, flat_I = sim.reshape(-1), ind.reshape(-1)
        sorted_sim_ids = (
            pd.DataFrame({"idx": flat_I, "score": flat_D})
            .sort_values(by="score", ascending=False)
            .drop_duplicates(subset="idx")
            .idx.tolist()
        )
        sim_meta = [ref_meta[ii] for ii in sorted_sim_ids][: args.topk]
        all_metas[i] = sim_meta
        ref_gps = [[item["LAT"], item["LON"]] for item in sim_meta]
        all_ref_gps.append(ref_gps)

    # compute metrics
    gt_gpses = np.array([[item["LAT"], item["LON"]] for item in df_test.to_dicts()])
    gt_labels = np.array([constant.LABELS2ID[i] for i in df_test["label"]])

    metrics = {
        "precision@10": precision_k(
            gt_gpses, gt_labels, all_metas, topk=10, min_dist=250
        ).item(),
        "precision@100": precision_k(
            gt_gpses, gt_labels, all_metas, topk=100, min_dist=250
        ).item(),
        "map_k@10": map_k(gt_gpses, gt_labels, all_metas, topk=10, min_dist=250).item(),
        "map_k@100": map_k(
            gt_gpses, gt_labels, all_metas, topk=100, min_dist=250
        ).item(),
    }

    print(metrics)

    with open("clip_retrieval_meta.json", "w") as f:
        f.write(json.dumps(all_metas))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--img-base-path", default=constant.IMAGE_BASE_PATH)
    parser.add_argument("--topk", type=int, default=constant.TOPK)
    parser.add_argument("--batch-size", type=int, default=constant.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=constant.SEED)
    parser.add_argument("--alpha-clip-name", default=constant.ALPHA_CLIP_MODEL_NAME)
    parser.add_argument("--mask-model-name", default=constant.MASKFORMER_MODEL_NAME)
    parser.add_argument(
        "--alpha-vision-ckpt", default=constant.ALPHA_CLIP_VISION_CHECKPOINT_PATH
    )
    args = parser.parse_args()

    main(args)
