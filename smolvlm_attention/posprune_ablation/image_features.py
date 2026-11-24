"""Build a temporary pre-connector cache from the pinned public DriveLM images."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from time import sleep
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
from PIL import Image
import requests
import torch
from tqdm.auto import tqdm

from smolvlm_inference.cache import CAMERA_ORDER, canonical, record_key


IMAGE_SOURCE = dict(repo='minhdang0901/drivelm-nuscenes-v1-0',
                    revision='6c4336a5279eaba3f3dfc5b525cc6ba57d55d61a')


def _image_path(value, camera):
    path = PurePosixPath(value.replace('\\', '/').removeprefix('data/'))
    camera_dir = 'CAM_' + camera.upper().replace('-', '_')
    if (len(path.parts) != 4 or path.parts[:3] != ('nuscenes', 'samples', camera_dir)
            or path.suffix.lower() not in ('.jpg', '.jpeg', '.png') or ':' in path.name):
        raise ValueError(f'Unexpected DriveLM image path for {camera}: {value}')
    return path.as_posix()


def frame_plan(rows):
    frames = {}
    for row in rows:
        key = record_key(row)
        if key not in frames:
            frames[key] = [_image_path(row['camera_paths'][camera], camera) for camera in CAMERA_ORDER]
    return sorted(frames.items())


def _fetch_image(filename):
    url = (f"https://huggingface.co/datasets/{IMAGE_SOURCE['repo']}/resolve/"
           f"{IMAGE_SOURCE['revision']}/{quote(filename, safe='/')}")
    for attempt in range(5):
        delay = 2 ** (attempt + 1)
        try:
            with requests.get(url, timeout=(12, 45)) as response:
                status = response.status_code
                if status == 200:
                    with Image.open(BytesIO(response.content)) as image:
                        return image.convert('RGB')
                if status not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f'Could not download {filename} from pinned DriveLM source (HTTP {status})')
                retry_after = response.headers.get('Retry-After', '')
                reset = re.search(r'(?:^|;)\s*t=(\d+)', response.headers.get('RateLimit', ''))
                if retry_after.isdecimal():
                    delay = max(delay, int(retry_after) + 1)
                if status == 429 and reset:
                    delay = max(delay, int(reset[1]) + 1)
                failure = f'HTTP {status}'
        except (requests.ConnectionError, requests.Timeout) as exc:
            failure = type(exc).__name__
        if attempt < 4:
            sleep(delay)
    raise RuntimeError(f'Could not download {filename} after retries ({failure})')


@torch.inference_mode()
def encode_frame(model, images):
    """Use native FP32 vision preprocessing and preserve the cache's FP16 storage format."""
    vision = model.native.model.vision_model
    device = next(vision.parameters()).device
    pixels = model.processor.image_processor(images=[images], return_tensors='pt')
    values = pixels['pixel_values'][0].to(device)
    masks = pixels['pixel_attention_mask'][0].to(device)
    patch = model.native.config.vision_config.patch_size
    patches = masks.unfold(1, patch, patch).unfold(2, patch, patch).any(-1).any(-1)
    hidden = vision(pixel_values=values, patch_attention_mask=patches).last_hidden_state
    features = hidden.to(device='cpu', dtype=torch.float16).numpy()
    geometry = model.geometry
    expected = (len(CAMERA_ORDER), geometry['vision_tokens'], geometry['vision_dim'])
    if features.shape != expected or not np.isfinite(features).all():
        raise ValueError(f'Unexpected vision features {features.shape}; expected finite values of shape {expected}')
    return features


def build_frames(model, frames, directory):
    """Download the six camera images concurrently for each unique assigned frame."""
    directory = Path(directory)
    with ThreadPoolExecutor(max_workers=len(CAMERA_ORDER)) as pool:
        for key, paths in tqdm(frames, desc='Download and encode DriveLM frames', unit='frame'):
            images = list(pool.map(_fetch_image, paths))
            try:
                features = encode_frame(model, images)
            finally:
                for image in images:
                    image.close()
            np.save(directory / f'{key}.npy', features, allow_pickle=False)


class _FrameArray:
    def __init__(self, directory):
        self.directory = Path(directory)

    def __getitem__(self, key):
        return np.load(self.directory / f'{key}.npy', allow_pickle=False, mmap_mode='r')


def image_cache(directory, frames, counts):
    fingerprint = sha256(canonical(dict(source=IMAGE_SOURCE, frame_keys=[key for key, _ in frames]))).hexdigest()
    return SimpleNamespace(array=_FrameArray(directory), index={key: key for key, _ in frames},
                           counts=counts, manifest=dict(fingerprint=fingerprint))
