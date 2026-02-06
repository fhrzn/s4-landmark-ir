import random

import numpy as np
import torch
from qdrant_client import QdrantClient, models


def haversine(origin: np.ndarray, destinations: np.ndarray) -> np.ndarray:
    """Compute great-circle distance in kilometers."""
    lat1, lon1 = np.radians(origin)
    lat2 = np.radians(destinations[:, 0])
    lon2 = np.radians(destinations[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return 6371.0 * c


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
