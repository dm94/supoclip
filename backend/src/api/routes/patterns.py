"""
API route for uploading pattern reference images for visual detection.
"""

import logging
from pathlib import Path
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from ...config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tasks/patterns", tags=["patterns"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@router.post("/upload")
async def upload_pattern_image(request: Request, file: UploadFile = File(...)):
    """
    Upload a reference image for visual pattern detection.

    Accepts PNG, JPG, JPEG, BMP files.
    Returne the local path for use in task creation.
    """
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    config = get_config()
    patterns_dir = Path(config.temp_dir) / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = patterns_dir / unique_name

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        dest.write_bytes(content)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save pattern image: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded image")

    logger.info("Pattern image saved: %s (%d bytes)", dest, len(content))

    return {
        "image_path": str(dest),
        "filename": unique_name,
    }
