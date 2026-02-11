import random

import numpy as np
import torch
from qdrant_client import QdrantClient, models


def haversine(gps1: list | tuple | np.ndarray, gps2: list | tuple | np.ndarray):
    if not isinstance(gps1, np.ndarray):
        gps1 = np.array(gps1)
    if not isinstance(gps2, np.ndarray):
        gps2 = np.array(gps2)

    gps1 = np.atleast_2d(gps1)
    gps2 = np.atleast_2d(gps2)

    lat1, lon1 = np.radians(gps1).T
    lat2, lon2 = np.radians(gps2).T

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    dist = 6371 * c
    if dist.size == 1:
        return dist.item()
    return dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


def create_collection(
    client: QdrantClient,
    name: str,
    vector_size: int = 768,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    try:
        client.get_collection(name)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
