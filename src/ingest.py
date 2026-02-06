import argparse
from uuid import uuid4

import polars as pl
from qdrant_client import QdrantClient, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import constant
from src.pipeline.feature_extractor import FeatureExtractor
from src.datasets.mp16 import MP16Dataset, collate_fn
from src.utils import get_device, set_seed, create_collection


def main(args):
    set_seed(args.seed)

    device = get_device()
    extractor = FeatureExtractor(
        alpha_clip_name=args.alpha_clip_name,
        mask_model_name=args.mask_model_name,
        alpha_vision_ckpt_pth=args.alpha_vision_ckpt,
        device=device,
    )

    df = pl.read_csv(args.data_path)
    dataset = MP16Dataset(
        df,
        img_col="IMG_ID",
        img_base_path=args.img_base_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    create_collection(client, args.collection_name)

    all_outputs = {}
    for batch, target, images in tqdm(loader, desc="Mask"):
        out = extractor(batch, images, target)
        for key, val in out.items():
            if key not in all_outputs:
                all_outputs[key] = val
            else:
                all_outputs[key].extend(val)

    for i in tqdm(range(len(df)), desc="ingest"):
        meta = df[i].to_dicts()[0]
        # meta["image"] = base64.b64encode(
        #     Image.fromarray(all_outputs["images"][i]).tobytes()
        # ).decode("utf-8")
        meta["index"] = i

        qdrant_points = []
        for sidx, (embed, mask, label) in tqdm(
            enumerate(zip(all_outputs["alpha_embeddings"][i], all_outputs["masks"][i], all_outputs["labels"][i])),
            desc="batch pack",
            leave=False,
        ):
            meta["segment_id"] = sidx
            meta["segment_mask"] = mask.tolist()
            meta["segment_label"] = label
            # meta["segment_mask_img"] = base64.b64encode(
            #     Image.fromarray(mask).tobytes()
            # ).decode("utf-8")
            qdrant_points.append(
                models.PointStruct(id=uuid4(), vector=embed, payload=meta)
            )

        client.upsert(collection_name=args.collection_name, points=qdrant_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Qdrant collection name. Auto-create if not exist yet.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="CSV with image metadata",
    )
    parser.add_argument(
        "--img-base-path",
        default=constant.IMAGE_BASE_PATH,
        help="Base path for images",
    )
    parser.add_argument("--qdrant-host", default=constant.QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=constant.QDRANT_PORT)
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
    args = parser.parse_args()

    main(args)
