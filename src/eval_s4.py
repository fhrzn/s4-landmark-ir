import argparse
import json
import os

import faiss
import numpy as np
import polars as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import constant
from src.datasets.mp16 import MP16Dataset, collate_fn
from src.geoclip import GeoCLIP
from src.metrics import map_k, precision_k
from src.pipeline.feature_extractor import FeatureExtractor
from src.utils import get_device, haversine, read_index, set_seed


def main(args):
    # prepare
    set_seed(args.seed)
    device = get_device()
    extractor = FeatureExtractor(
        alpha_clip_name=args.alpha_clip_name,
        mask_model_name=args.mask_model_name,
        alpha_vision_ckpt_pth=args.alpha_vision_ckpt,
        device=device,
    )
    geoclip_model = GeoCLIP().to(device)

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
        collate_fn=collate_fn,
    )
    index, ref_meta = read_index(args.index_dir)
    index.hnsw.efSearch = 128
    ref_meta = ref_meta["metadata"]

    # mask
    all_outputs = {}
    for batch, target, images in tqdm(loader, desc="Mask"):
        out = extractor(batch, images, target)
        for key, val in out.items():
            if key not in all_outputs:
                all_outputs[key] = val
            else:
                all_outputs[key].extend(val)

    # predict gps for reranking later
    pred_gps = []
    img_paths = df_test["IMG_ID"].to_list()
    for i in tqdm(range(0, len(df_test), args.batch_size), desc="predict gps"):
        paths = img_paths[i : i + args.batch_size]
        paths = [os.path.join("./datasets/mp16-reason", p) for p in paths]
        _gps, _ = geoclip_model.predict_batch(paths, top_k=1)
        _gps = _gps.view(-1, 2)
        pred_gps.extend(_gps.tolist())

    # similarity search
    all_ref_gps = []
    all_metas = {}
    for i in tqdm(range(len(df_test)), desc="eval"):
        query_emb = all_outputs["alpha_embeddings"][i].astype("float32")
        faiss.normalize_L2(query_emb)
        sim, ind = index.search(query_emb, args.topk)
        _, flat_I = sim.reshape(-1), ind.reshape(-1)
        sim_meta = [ref_meta[ii] for ii in flat_I]
        sim_meta = (
            pl.DataFrame([ref_meta[ii] for ii in flat_I])
            .unique(subset=["idx"])
            .to_dicts()
        )
        ref_gps = [[item["LAT"], item["LON"]] for item in sim_meta]

        # reranking
        ranks = np.argsort(haversine(pred_gps[i], ref_gps))[::-1]
        sim_meta = [sim_meta[i] for i in ranks]
        ref_gps = [ref_gps[i] for i in ranks]

        all_metas[i] = sim_meta
        all_ref_gps.append(ref_gps)

    all_ref_gps = np.array(all_ref_gps)

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

    with open("s4_retrieval_meta.json", "w") as f:
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
