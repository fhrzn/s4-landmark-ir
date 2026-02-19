import polars as pl
import randomname
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mp16 import MP16Dataset, collate_fn
from src.pipeline.feature_extractor import FeatureExtractor
from src.utils import add_record_to_index, build_index, save_index


def ingest(args):

    index = build_index(args.index_size)

    df = pl.read_csv(args.data_path)
    dataset = MP16Dataset(df, img_col="IMG_ID", img_base_path=args.img_base_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    extractor = FeatureExtractor(
        alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit20m_fultune_2xe.pth"
    )
    all_outputs = {}
    for batch, target, images in tqdm(loader, desc="Mask"):
        out = extractor(batch, images, target)
        for key, val in out.items():
            if key not in all_outputs:
                all_outputs[key] = val
            else:
                all_outputs[key].extend(val)

    metadatas = []
    for i, ref_item in tqdm(
        enumerate(df.to_dicts()), total=len(df), desc="ingest to index"
    ):
        for n in range(len(all_outputs["labels"][i])):
            metadatas.append(ref_item)

        _embed = all_outputs["alpha_embeddings"][i].astype("float32")
        add_record_to_index(index, _embed)

    target_dir = args.output_dir if args.output_dir else randomname.generate(sep="_")
    save_index(index, metadatas, target_dir=target_dir)

    print(f"index and metadata saved successfully to {target_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--img-base-path", default="./datasets/mp16-reason")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--index-size", type=int, default=768)
    parser.add_argument("--output-dir")

    args = parser.parse_args()

    ingest(args)
