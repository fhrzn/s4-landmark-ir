from typing import List

import alpha_clip
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        alpha_clip_name: str = "ViT-L/14",
        mask_model_name: str = "facebook/mask2former-swin-large-ade-semantic",
        alpha_vision_ckpt_pth: str = "../checkpoints/clip_l14_grit20m_fultune_2xe.pth",
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        self.device = torch.device(device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            self.mask_processor = AutoImageProcessor.from_pretrained(mask_model_name)
            self.mask_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                mask_model_name
            ).to(self.device)
            self.mask_model = self.mask_model.eval()

        self.alphaclip_model, self.alphaclip_processor = alpha_clip.load(
            alpha_clip_name, alpha_vision_ckpt_pth=alpha_vision_ckpt_pth, device=self.device
        )
        transform_size = 336 if alpha_clip_name == "ViT-L/14@336px" else 224
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((transform_size, transform_size)),
                transforms.Normalize(0.5, 0.26),
            ]
        )

    @torch.no_grad()
    def forward(
        self, batch_inputs: torch.Tensor, images: List[Image.Image], target_sizes
    ):
        # mask generation
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            outputs = self.mask_model(**batch_inputs)

        segs = self.mask_processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )

        alpha_embeddings = []
        all_masks = []
        all_labels = []
        for seg, img in zip(segs, images):
            seg_np = seg.cpu().numpy().astype(np.int32)
            mask_indices = np.unique(seg_np)
            alpha = []
            masks = []
            labels = []
            for label in mask_indices:
                mask = seg_np == label
                mask = (mask * 255).astype(np.uint8)
                masks.append(mask)
                mask = self.mask_transform(mask)
                alpha.append(mask)
                labels.append(self.mask_model.config.id2label[label])
            all_masks.append(masks)
            all_labels.append(labels)

            # mask embedding
            alpha_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            alpha = torch.stack(alpha).to(self.device, dtype=alpha_dtype)
            alpha_img_processed = self.alphaclip_processor(img).unsqueeze(0).to(
                self.device, dtype=alpha_dtype
            )
            alpha_out = (
                self.alphaclip_model.visual(alpha_img_processed, alpha).cpu().numpy()
            )
            alpha_embeddings.append(alpha_out)

            del alpha_out

        del outputs
        del segs
        torch.cuda.empty_cache()

        return {
            "alpha_embeddings": alpha_embeddings,
            "masks": all_masks,
            "labels": all_labels,
            "images": [np.array(img) for img in images],
        }
