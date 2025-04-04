# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    def translate(img: np.ndarray, shift_x: int = 30, shift_y: int = 30) -> np.ndarray:

        translated = np.zeros_like(img)
        max_y, max_x = img.shape
        shift_y = min(shift_y, max_y)
        shift_x = min(shift_x, max_x)
        translated[shift_y:, shift_x:] = img[:max_y-shift_y, :max_x-shift_x]
        return translated
    
    def rotate_90_clockwise(img: np.ndarray) -> np.ndarray:
        return np.flipud(np.transpose(img))
    
    def stretch_horizontal(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
        h, w = img.shape
        new_w = int(w * scale)
        x_new = np.linspace(0, w-1, new_w)
        x_floor = np.floor(x_new).astype(int)
        x_ceil = np.ceil(x_new).astype(int)
        alpha = x_new - x_floor
        
        x_floor = np.clip(x_floor, 0, w-1)
        x_ceil = np.clip(x_ceil, 0, w-1)
        
        stretched = (1-alpha) * img[:, x_floor] + alpha * img[:, x_ceil]
        return stretched.astype(img.dtype)
    
    def mirror_horizontal(img: np.ndarray) -> np.ndarray:
        return np.fliplr(img)
    
    def barrel_distort(img: np.ndarray, strength: float = 0.1) -> np.ndarray:
        h, w = img.shape
        y, x = np.indices((h, w))
        cx, cy = w // 2, h // 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(cx, cy)
        factor = 1 + strength * r**2
        
        new_x = np.clip(((x - cx) * factor + cx).astype(int), 0, w - 1)
        new_y = np.clip(((y - cy) * factor + cy).astype(int), 0, h - 1)
        return img[new_y, new_x]
    
    return {
        "translated": translate(img),
        "rotated": rotate_90_clockwise(img),
        "stretched": stretch_horizontal(img),
        "mirrored": mirror_horizontal(img),
        "distorted": barrel_distort(img)
    }
    pass

# resto do codigo 