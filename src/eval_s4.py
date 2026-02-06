import argparse
from typing import Iterable, List, Optional

import numpy as np
import polars as pl
from PIL import Image
from qdrant_client import QdrantClient, models

import constant
from src.pipeline.feature_extractor import FeatureExtractor
from src.utils import get_device, set_seed
from src.utils import haversine



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

    return [
        {**merged_response[i], "distance": distances[i].item()} for i in rank_desc
    ]


def main(args):
    set_seed(args.seed)

    device = get_device()
    extractor = FeatureExtractor(
        alpha_clip_name=args.alpha_clip_name,
        mask_model_name=args.mask_model_name,
        alpha_vision_ckpt_pth=args.alpha_vision_ckpt,
        device=device,
    )

    query_img = Image.open(args.query_image).convert("RGB")
    query_img_size = [query_img.size[::-1]]
    query_input = extractor.mask_processor(images=query_img, return_tensors="pt")
    query_embed = extractor(query_input, [query_img], query_img_size)

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    query_res = query_qdrant(
        client,
        args.collection_name,
        query_embed["alpha_embeddings"][0],
        args.limit,
    )

    merged_response = merge_responses(query_res)

    if args.reference_csv:
        df_ref = pl.read_csv(args.reference_csv)
        merged_response = maybe_rerank(
            merged_response, df_ref, args.rerank_gt_lat, args.rerank_gt_lon
        )

    print(f"Segments queried: {len(query_res)}")
    print(f"Unique results: {len(merged_response)}")
    for idx, payload in enumerate(merged_response[: args.limit]):
        city = payload.get("city", ["?"])[0]
        country = payload.get("country", ["?"])[0]
        img_id = payload.get("IMG_ID", [""])[0]
        print(f"{idx+1}. {city}, {country} | {img_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-name", required=True, help="Qdrant collection"
    )
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
    parser.add_argument("--limit", type=int, default=10)
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
