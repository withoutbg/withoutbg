"""FastAPI backend for withoutbg web application."""

import io
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps 
import pillow_heif  # Required for HEIF/HEIC decoding
import uvicorn

# Enable HEIF support globally for Pillow
pillow_heif.register_heif_opener()

# Import withoutbg package (install via: uv sync or pip install -e ../../../packages/python)
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


@app.on_event("startup")
async def startup_event():
    """Initialize models at startup for optimal performance."""
    global _model
    print("Loading Open Source models...")
    _model = WithoutBG.opensource()
    print("✓ Models loaded and ready for inference!")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "service": "withoutbg-api",
        "models_loaded": _model is not None
    }


@app.post("/api/remove-background")
async def remove_background_endpoint(
    file: UploadFile = File(...),
    format: str = Form("png"),
    quality: int = Form(95),
    api_key: Optional[str] = Form(None),
):
    """
    Remove background from a single image.
    
    Args:
        file: Image file to process
        format: Output format (png, jpg, webp)
        quality: Quality for JPEG output (1-100)
        api_key: Optional API key for cloud processing
    
    Returns:
        Processed image with background removed
    """
    try:
        # Support standard image types and native Apple HEIC/HEIF
        is_image = file.content_type and file.content_type.startswith("image/")
        is_heic = file.filename and file.filename.lower().endswith((".heic", ".heif"))
        
        if not (is_image or is_heic):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, or HEIC)")
        
        # Read uploaded file
        contents = await file.read()
        raw_image = Image.open(io.BytesIO(contents))
        
        # 1. Apply EXIF orientation (prevents rotated mobile uploads)
        # 2. Force RGBA for consistency across inference models
        input_image = ImageOps.exif_transpose(raw_image).convert("RGBA")
        
        # Process image using appropriate model
        if api_key:
            # Use API for this specific request
            api_model = WithoutBG.api(api_key)
            result = api_model.remove_background(input_image)
        else:
            # Use pre-loaded opensource model (fast!)
            if _model is None:
                raise HTTPException(
                    status_code=503,
                    detail="Models not loaded. Server may still be starting up."
                )
            result = _model.remove_background(input_image)
        
        # Convert result to bytes
        output_buffer = io.BytesIO()
        
        # Handle format conversion
        if format.lower() in ["jpg", "jpeg"]:
            # Convert RGBA to RGB for JPEG
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
        else:  # PNG
            result.save(output_buffer, format="PNG")
            media_type = "image/png"
        
        output_buffer.seek(0)
        
        return Response(
            content=output_buffer.getvalue(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"inline; filename=withoutbg.{format}"
            }
        )
        
    except WithoutBGError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/usage")
async def get_usage_endpoint(api_key: str):
    """
    Get API usage statistics.
    
    Args:
        api_key: API key for cloud service
    
    Returns:
        Usage statistics
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
    async def root():
        """Serve the React frontend index.html at root."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not found")
    
    # Catch-all route for React SPA - must be last
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React frontend for all non-API routes."""
        # Don't serve frontend for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Try to serve the requested file
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        
        # Otherwise, serve index.html (SPA routing)
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        
        raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
