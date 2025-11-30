import base64
import io
import os
import requests
from PIL import Image
import numpy as np
import torch


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to torch tensor in shape [1, 3, H, W], float32 [0..1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:  # grayscale -> replicate channels
        arr = np.stack([arr, arr, arr], axis=-1)
    # H, W, C -> C, H, W
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr)[None, ...]


class NanoBananaChat:
    """
    ComfyUI custom node that sends a prompt to a Nano Banana-like image API
    and returns an IMAGE tensor for ComfyUI.

    Usage:
      - Put this file in your ComfyUI `custom_nodes/` folder.
      - Configure `API_URL` and `API_KEY` here or via environment variables
        `NANOBANANA_API_URL` and `NANOBANANA_API_KEY`.
    """

    def __init__(self):
        # Allow overriding via env vars
        self.API_URL = os.environ.get("NANOBANANA_API_URL", self.__class__.API_URL)
        self.API_KEY = os.environ.get("NANOBANANA_API_KEY", self.__class__.API_KEY)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a heroic warrior in dark fantasy armor, dramatic lighting",
                }),
                "aspect_ratio": ("STRING", {
                    "default": "16:9",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "NanoBanana"

    # --- ВАЖНО ---
    # По умолчанию заполните эти поля или задайте переменные окружения:
    API_URL = "https://YOUR_NANOBANANA_ENDPOINT"
    API_KEY = "YOUR_API_KEY_HERE"

    def call_nanobanana_api(self, prompt: str, aspect_ratio: str) -> Image.Image:
        """
        Example API call. Проверьте документацию вашего провайдера и
        скорректируйте заголовки/тело запроса и разбор ответа соответственно.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}",
        }

        data = {
            "prompt": prompt,
            "image_size": aspect_ratio,
        }

        resp = requests.post(self.API_URL, json=data, headers=headers, timeout=120)
        resp.raise_for_status()
        result = resp.json()

        # Пример: API возвращает base64 в поле `image_base64`.
        # Отредактируйте в соответствии с реальным ответом API.
        if "image_base64" in result:
            b64 = result["image_base64"]
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return img

        # Некоторые провайдеры возвращают массив байт в `data` или ссылку.
        # Попробуем скопом обработать несколько вариантов:
        if "image_url" in result:
            img_resp = requests.get(result["image_url"], timeout=60)
            img_resp.raise_for_status()
            img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            return img

        raise ValueError("Неизвестный формат ответа от NanoBanana API: " + str(result))

    def generate(self, prompt, aspect_ratio):
        img = self.call_nanobanana_api(prompt, aspect_ratio)
        tensor = pil_to_tensor(img)
        return (tensor,)


NODE_CLASS_MAPPINGS = {
    "NanoBananaChat": NanoBananaChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaChat": "Nano Banana Chat",
}
