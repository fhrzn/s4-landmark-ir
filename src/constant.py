MASKFORMER_MODEL_NAME = "facebook/mask2former-swin-large-ade-semantic"
ALPHA_CLIP_MODEL_NAME = "ViT-L/14"
ALPHA_CLIP_VISION_CHECKPOINT_PATH = "./checkpoints/clip_l14_grit20m_fultune_2xe.pth"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
BATCH_SIZE = 64
SEED = 42
IMAGE_BASE_PATH = "./datasets/mp16-reason"
TOPK = 100
HF_VLM_NAME = "Qwen/Qwen3-VL-2B-Instruct"

LABELS2ID = {
    "religious_site": 0,
    "historic_civic": 1,
    "monument_memorial": 2,
    "tower": 3,
    "bridge": 4,
    "gate": 5,
    "museum_culture": 6,
    "leisure_park": 7,
    "urban_scene": 8,
    "natural": 9,
    "unknown": -1,
}