from typing import List

from pydantic import BaseModel, Field


class TextToImageResponse(BaseModel):
    images: List[str] = Field(
        default=None, title="Image", description="The generated image in base64 format."
    )
