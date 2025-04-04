# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""



import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
import os


def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    if i1.shape != i2.shape:
        raise ValueError("As imagens devem ter as mesmas dimensões")
    
    def mse(i1: np.ndarray, i2: np.ndarray) -> float:
        return np.mean((i1 - i2) ** 2)
    
    def psnr(i1: np.ndarray, i2: np.ndarray) -> float:
        mse_value = mse(i1, i2)
        return float('inf') if mse_value == 0 else 10 * np.log10(1.0 / mse_value)
    
    def ssim(i1: np.ndarray, i2: np.ndarray) -> float:
        k1, k2 = 0.01, 0.03
        L = 1.0 
        c1, c2 = (k1*L)**2, (k2*L)**2
        
        mean1, mean2 = np.mean(i1), np.mean(i2)
        var1, var2 = np.var(i1), np.var(i2)
        covar = np.cov(i1.flatten(), i2.flatten())[0, 1]
        
        luminance = (2*mean1*mean2 + c1) / (mean1**2 + mean2**2 + c1)
        contrast = (2*np.sqrt(var1)*np.sqrt(var2) + c2) / (var1 + var2 + c2)
        structure = (covar + c2/2) / (np.sqrt(var1)*np.sqrt(var2) + c2/2)
        
        return luminance * contrast * structure
    
    def npcc(i1: np.ndarray, i2: np.ndarray) -> float:
        cov = np.cov(i1.flatten(), i2.flatten())[0, 1]
        std1, std2 = np.std(i1), np.std(i2)
        return cov / (std1 * std2) if (std1 * std2) != 0 else 0
    
    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }

    pass
# --- Função de teste 