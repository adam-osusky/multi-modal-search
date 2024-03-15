import base64
from io import BytesIO

from PIL import Image


def get_img_base64(filepath: str):
    pil_image = Image.open(filepath)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
