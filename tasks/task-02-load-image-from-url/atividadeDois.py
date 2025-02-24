import argparse
import numpy as np
import cv2 as cv


def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    import requests 
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers, stream=True)
   
    if response.status_code != 200:
       raise ValueError(f"Erro ao baixar a imagem {response.status_code}")

    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    image = cv.imdecode(image_array, flags=kwargs.get("flags",  cv.IMREAD_GRAYSCALE )) 
     #  cv.IMREAD_UNCHANGED cv.IMREAD_GRAYSCALE  cv.IMREAD_ANYCOLOR cv.IMREAD_COLOR

    if image is None:
        raise ValueError("Falha ao decodificar a imagem.")

    return image

load_image_from_url() 

