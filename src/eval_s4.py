import argparse
from typing import Iterable, List, Optional

import numpy as np
import polars as pl
from qdrant_client import QdrantClient, models

from src import constant
from src.pipeline.feature_extractor import FeatureExtractor
from src.utils import get_device, set_seed
from src.utils import haversine
from src.datasets.mp16 import MP16Dataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm


def main(args):
    set_seed(args.seed)

    device = get_device()
    extractor = FeatureExtractor(
        alpha_clip_name=args.alpha_clip_name,
        mask_model_name=args.mask_model_name,
        alpha_vision_ckpt_pth=args.alpha_vision_ckpt,
        device=device,
    )

    df_ref = pl.read_csv(args.reference_path)
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

    all_outputs = {}
    for batch, target, images in tqdm(loader, desc="Mask"):
        out = extractor(batch, images, target)
        for key, val in out.items():
            if key not in all_outputs:
                all_outputs[key] = val
            else:
                all_outputs[key].extend(val)
        break

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    for i in tqdm(range(len(df_test)), desc="eval"):
        # query
        search_queries = [
            models.QueryRequest(query=embed, limit=args.topk, with_payload=True)
            for embed in all_outputs["alpha_embeddings"][i]
        ]
        query_res = client.query_batch_points(
            collection_name=args.collection_name, requests=search_queries
        )
        query_res = merge_responses(query_res)
        
        # rerank
        ref_ids = [d["index"] for d in query_res]
        ref_gps = df_ref[ref_ids][:, ["LAT", "LON"]].to_numpy()
        gt_gps = df_test[i]["LAT", "LON"].to_numpy().reshape(-1)

        distances = haversine(gt_gps, ref_gps)
        rank = np.argsort(distances)[::-1]
        reranked_response = [
            {**query_res[i], "distance": distances[i].item()}
            for i in rank
        ]

        # TODO: eval metrics

        break


def query_qdrant(
    client: QdrantClient,
    collection_name: str,
    embeddings: Iterable[np.ndarray],
    limit: int,
):
    search_queries = [
        models.QueryRequest(query=embed, limit=limit, with_payload=True)
        for embed in embeddings
    ]
    return client.query_batch_points(
        collection_name=collection_name, requests=search_queries
    )


def merge_responses(query_res) -> List[dict]:
    merged_response = {}
    for qres in query_res:
        for item in qres.points:
            payload = item.payload
            if payload.get("index") not in merged_response:
                merged_response[payload.get("index")] = payload
    return list(merged_response.values())


def maybe_rerank(
    merged_response: List[dict],
    df_ref: pl.DataFrame,
    gt_lat: Optional[float],
    gt_lon: Optional[float],
) -> List[dict]:
    if gt_lat is None or gt_lon is None:
        return merged_response

    id_responses = [d["index"] for d in merged_response]
    gps_responses = df_ref[id_responses][:, ["LAT", "LON"]].to_numpy()
    distances = haversine(np.array((gt_lat, gt_lon)), gps_responses)
    rank_desc = np.argsort(distances)[::-1]

    return [{**merged_response[i], "distance": distances[i].item()} for i in rank_desc]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-name", required=True, help="Qdrant collection")
    parser.add_argument(
        "--data-path",
        required=True,
        help="CSV with image metadata",
    )
    parser.add_argument(
        "--reference-path",
        required=True,
        help="Optional CSV for reference metadata",
    )
    parser.add_argument(
        "--img-base-path",
        default=constant.IMAGE_BASE_PATH,
        help="Base path for images",
    )
    parser.add_argument("--qdrant-host", default=constant.QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=constant.QDRANT_PORT)
    parser.add_argument("--topk", type=int, default=constant.TOPK)
    parser.add_argument("--batch-size", type=int, default=constant.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=constant.SEED)
    parser.add_argument(
        "--alpha-clip-name",
        default=constant.ALPHA_CLIP_MODEL_NAME,
        help="AlphaCLIP model name",
    )
    parser.add_argument(
        "--mask-model-name",
        default=constant.MASKFORMER_MODEL_NAME,
        help="Mask2Former model name",
    )
    parser.add_argument(
        "--alpha-vision-ckpt",
        default=constant.ALPHA_CLIP_VISION_CHECKPOINT_PATH,
        help="AlphaCLIP vision checkpoint path",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=True,
        help="Use reranker or not",
    )
    args = parser.parse_args()

    main(args)
