import base64
from io import BytesIO


def encode_pil_to_base64(pil_image):
    image_buffer = BytesIO()
    pil_image.save(image_buffer, 'JPEG')
    image_buffer.seek(0)

    base64_string = "data:image/jpeg;base64," + base64.b64encode(image_buffer.getvalue()).decode()

    return base64_string
