"""FastAPI backend for withoutbg web application."""

import io
import os
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# Import withoutbg package
# (install via: uv sync or pip install -e ../../../packages/python)
from withoutbg import WithoutBG, __version__
from withoutbg.exceptions import WithoutBGError
from withoutbg.api import ProAPI

app = FastAPI(
    title="withoutbg API",
    description="AI-powered background removal API",
    version=__version__,
)

# CORS middleware for local development and deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # Expose header for blob downloads
)

# Static files directory (frontend build)
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

# Global model instance (initialized at startup, reused for all requests)
_model: Optional[WithoutBG] = None

# Allowed image MIME types for URL fetching
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}

# Max size for URL-fetched images (20 MB)
MAX_IMAGE_BYTES = 20 * 1024 * 1024


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize models at startup for optimal performance."""
    global _model
    print("Loading Open Source models...")
    _model = WithoutBG.opensource()
    print("✓ Models loaded and ready for inference!")


@app.get("/api/health")
async def health_check() -> dict[str,object]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "service": "withoutbg-api",
        "models_loaded": _model is not None,
    }


async def fetch_image_from_url(url: str) -> Image.Image:
    """
    Fetch an image from a URL and return it as a PIL Image.

    Args:
        url: Public URL of the image to fetch.

    Returns:
        PIL Image object.

    Raises:
        HTTPException: On network errors, non-image content, or oversized responses.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=408, detail="Request timed out while fetching image URL."
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch image from URL (HTTP {e.response.status_code}).",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to reach image URL: {str(e)}"
        )

    # Validate content type
    content_type = (
        response.headers.get("content-type", "").split(";")[0].strip().lower()
    )
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"URL does not point to a supported image (got '{content_type}'). "
            f"Supported types: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}.",
        )

    # Guard against huge payloads
    if len(response.content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image from URL exceeds the "
                f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB size limit."
            ),
        )

    try:
        return Image.open(io.BytesIO(response.content))
    except Exception:
        raise HTTPException(
            status_code=400, detail="URL content could not be decoded as an image."
        )


@app.post("/api/remove-background")
async def remove_background_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    format: str = Form("png"),
    quality: int = Form(95),
    api_key: Optional[str] = Form(None),
) -> Response:
    """
    Remove background from a single image.

    Provide either `file` (a direct upload) **or** `image_url` (a publicly
    accessible URL). If both are supplied, `file` takes precedence.

    Args:
        file: Image file to process (multipart upload).
        image_url: Public URL of an image to fetch and process.
        format: Output format — png (default), jpg, or webp.
        quality: Quality for JPEG/WebP output (1-100, default 95).
        api_key: Optional API key to use cloud processing instead of the
                 local open-source model.

    Returns:
        Processed image with background removed.
    """
    # ── 1. Resolve input image ────────────────────────────────────────────────
    if file is not None:
        # Direct upload path (original behaviour)
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Uploaded file must be an image."
            )
        contents = await file.read()
        try:
            input_image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file could not be decoded as an image.",
            )

    elif image_url:
        # URL input path (new)
        input_image = await fetch_image_from_url(image_url)

    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either a file or an image URL.",
        )

    # ── 2. Run background removal ─────────────────────────────────────────────
    try:
        if api_key:
            # Use API for this specific request
            api_model = WithoutBG.api(api_key)
            result = api_model.remove_background(input_image)
        else:
            # Use pre-loaded opensource model (fast!)
            if _model is None:
                raise HTTPException(
                    status_code=503,
                    detail="Models not loaded. Server may still be starting up.",
                )
            result = _model.remove_background(input_image)

    except WithoutBGError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # ── 3. Encode output ──────────────────────────────────────────────────────
    output_buffer = io.BytesIO()

    if format.lower() in ["jpg", "jpeg"]:
        if result.mode == "RGBA":
            rgb_image = Image.new("RGB", result.size, (255, 255, 255))
            rgb_image.paste(result, mask=result.split()[3])
            rgb_image.save(output_buffer, format="JPEG", quality=quality)
        else:
            result.save(output_buffer, format="JPEG", quality=quality)
        media_type = "image/jpeg"
    elif format.lower() == "webp":
        result.save(output_buffer, format="WEBP", quality=quality)
        media_type = "image/webp"
    else:  # PNG (default)
        result.save(output_buffer, format="PNG")
        media_type = "image/png"

    output_buffer.seek(0)

    return Response(
        content=output_buffer.getvalue(),
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename=withoutbg.{format}"},
    )


@app.get("/api/usage")
async def get_usage_endpoint(api_key: str) -> JSONResponse:
    """
    Get API usage statistics.

    Args:
        api_key: API key for cloud service.

    Returns:
        Usage statistics.
    """
    try:
        api = ProAPI(api_key)
        usage = api.get_usage()
        return JSONResponse(content=usage)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch usage: {str(e)}")


# Mount static files (only if directory exists - for production)
if STATIC_DIR.exists():
    # Serve static assets (js, css, images, etc.)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Root route - serve index.html
    @app.get("/")
    async def root() -> FileResponse:
        """Serve the React frontend index.html at root."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not found")

    # Catch-all route for React SPA - must be last
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str) -> FileResponse:
        """Serve the React frontend for all non-API routes."""
        # Don't serve frontend for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

        raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
