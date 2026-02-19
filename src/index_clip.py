from functools import partial

import polars as pl
import randomname
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from src.datasets.mp16 import MP16Dataset
from src.utils import add_record_to_index, build_index, get_device, save_index, clip_collate_fn


def ingest(args):
    # prepare
    index = build_index(args.index_size)

    device = get_device()
    clip_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model = clip_model.eval()
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # dataset
    df = pl.read_csv(args.data_path)
    dataset = MP16Dataset(df, img_col="IMG_ID", img_base_path=args.img_base_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(clip_collate_fn, clip_processor),
    )

    # encode
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            out = clip_model.get_image_features(
                **{k: v.to(device) for k, v in batch.items()}
            )
            out = out.pooler_output.cpu().numpy()
            add_record_to_index(index, out)

    metadatas = df.to_dicts()

    # save
    target_dir = args.output_dir if args.output_dir else randomname.generate(sep="_")
    save_index(index, metadatas, target_dir=target_dir)

    print(f"index and metadata saved successfully to {target_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--img-base-path", default="./datasets/mp16-reason")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-size", type=int, default=768)
    parser.add_argument("--output-dir")

    args = parser.parse_args()

    ingest(args)
