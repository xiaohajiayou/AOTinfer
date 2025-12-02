"""
OpenCV-based image preprocessing to mimic MiniCPM HF processor:
 - optional slicing into grid tiles when image is large
 - resize to scale_resolution (default 448) with dimensions divisible by patch_size
 - normalize (mean/std=0.5), convert to CHW

Outputs:
 - pixel_values: list of np.ndarray per slice, shape [3, H, W] (normalized, RGB)
 - tgt_sizes: np.ndarray of shape [num_slices, 2] with patch grid (H/patch, W/patch)
 - slice_counts: number of slices per original image

Note: image_bound (占位符起止) 依赖文本 token 位置，这里仅返回每张图的切片数，供上游根据模板计算占位长度（每切片 64）。
"""

from typing import List, Tuple
import torch
import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("opencv-python is required for image preprocessing") from e


def ensure_divide(length: int, patch_size: int) -> int:
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size: Tuple[int, int], scale_resolution: int, patch_size: int, allow_upscale: bool) -> Tuple[int, int]:
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / np.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return best_width, best_height


def get_sliced_grid(image_size: Tuple[int, int], max_slice_nums: int, scale_resolution: int, patch_size: int) -> Tuple[int, int] | None:
    original_width, original_height = image_size
    log_ratio = np.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(int(np.ceil(ratio)), max_slice_nums)
    if multiple <= 1:
        return None
    candidate_split_nums = []
    for i in [multiple - 1, multiple, multiple + 1]:
        if i == 1 or i > max_slice_nums:
            continue
        candidate_split_nums.append(i)
    candidate_grids = []
    for split_num in candidate_split_nums:
        for m in range(1, split_num + 1):
            if split_num % m == 0:
                candidate_grids.append((m, split_num // m))
    best_grid = (1, 1)
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - np.log(grid[0] / grid[1]))
        if error < min_error:
            best_grid = grid
            min_error = error
    return best_grid


def split_to_patches(image: np.ndarray, grid: Tuple[int, int]) -> List[np.ndarray]:
    """
    Args:
        image: numpy array (H,W,3) in RGB
        grid: (cols, rows)
    Returns:
        list of patch images (H_i,W_i,3) in RGB
    """
    patches = []
    width, height = image.shape[1], image.shape[0]
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])
    for i in range(0, height, grid_y):
        for j in range(0, width, grid_x):
            box = image[i : i + grid_y, j : j + grid_x, :]
            patches.append(box)
    return patches


def slice_image(
    image: np.ndarray,
    max_slice_nums: int,
    scale_resolution: int,
    patch_size: int,
) -> List[np.ndarray]:
    """Return list of slice images (RGB). Includes original resized image as the first element."""
    original_size = (image.shape[1], image.shape[0])  # (W,H)
    slices = []
    grid = get_sliced_grid(original_size, max_slice_nums, scale_resolution, patch_size)
    if grid is None:
        best_size = find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
        src = cv2.resize(image, best_size, interpolation=cv2.INTER_CUBIC)
        slices.append(src)
        return slices
    # downsample original, ensure divisible
    best_resize = find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False)
    src = cv2.resize(image, best_resize, interpolation=cv2.INTER_CUBIC)
    slices.append(src)
    # refine size to grid and slice
    refine_width = ensure_divide(original_size[0], grid[0])
    refine_height = ensure_divide(original_size[1], grid[1])
    refine_image = cv2.resize(image, (refine_width, refine_height), interpolation=cv2.INTER_CUBIC)
    slices.extend(split_to_patches(refine_image, grid))
    return slices


def normalize_and_chw(
    img_rgb: np.ndarray,
    patch_size: int,
    target_size: int,
    mean: float = 0.5,
    std: float = 0.5,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Args:
        img_rgb: np.ndarray uint8 or float, shape (H,W,3), RGB
    Returns:
        chw: np.ndarray [3, H, W] normalized
        grid: (H/patch, W/patch)
    """
    # 强制 resize 到目标尺寸，保持与 HF vision_config.image_size 对齐
    if img_rgb.shape[0] != target_size or img_rgb.shape[1] != target_size:
        import cv2

        img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    img = img_rgb.astype(np.float32) / 255.0
    img = (img - mean) / std
    # to CHW
    chw = np.transpose(img, (2, 0, 1))
    C, H, W = chw.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H/W must be divisible by patch_size"
    grid = (H // patch_size, W // patch_size)
    return chw, grid


def preprocess_images(
    images: List[np.ndarray],
    max_slice_nums: int = 9,
    scale_resolution: int = 448,
    patch_size: int = 14,
) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    """
    Args:
        images: list of images in RGB np.ndarray (H,W,3)
    Returns:
        pixel_values: list of np.ndarray [3, patch_size, num_patches] per slice
        tgt_sizes: np.ndarray [num_slices, 2]
        slice_counts: list of slice counts per image
    """
    pixel_values = []
    tgt_sizes = []
    slice_counts = []
    for img in images:
        slices = slice_image(img, max_slice_nums, scale_resolution, patch_size)
        slice_counts.append(len(slices))
        for s in slices:
            chw, grid = normalize_and_chw(s, patch_size, target_size=scale_resolution)
            pixel_values.append(chw)
            tgt_sizes.append(np.array(grid, dtype=np.int32))
    tgt_sizes_arr = np.vstack(tgt_sizes) if tgt_sizes else np.zeros((0, 2), dtype=np.int32)
    return pixel_values, tgt_sizes_arr, slice_counts


def build_image_bound_from_lengths(start_idx: int, lengths: List[int], token_per_slice: int = 64) -> np.ndarray:
    """
    Utility: given a start token index and per-image slice counts, build image_bound entries.
    """
    bounds = []
    cur = start_idx
    for l in lengths:
        end = cur + l * token_per_slice
        bounds.append([cur, end])
        cur = end
    return np.array(bounds, dtype=np.int64)


def pad_slices_to_tensor(
    pixel_values: List[np.ndarray],
    tgt_sizes: np.ndarray,
    s_max: int,
    h: int,
    w: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad/stack slice outputs to fixed shapes:
      pixel_values -> [S_max, 3, H, W]
      tgt_sizes -> [S_max, 2]
    """
    import torch

    S = min(len(pixel_values), s_max)
    pv = torch.zeros((s_max, 3, h, w), dtype=torch.float32, device=device)
    ts = torch.zeros((s_max, 2), dtype=torch.int64, device=device)
    for i in range(S):
        chw = pixel_values[i]  # [3, H, W]
        h_i, w_i = chw.shape[1], chw.shape[2]
        pv[i, :, :h_i, :w_i] = torch.from_numpy(chw)
        ts[i] = torch.from_numpy(tgt_sizes[i])
    return pv, ts
