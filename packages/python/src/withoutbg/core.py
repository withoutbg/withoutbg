"""Core background removal functionality with class-based API."""

import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

from PIL import Image

from .api import ProAPI
from .exceptions import WithoutBGError
from .models import OpenSourceModel


class WithoutBG:
    """Base class for background removal.

    Use factory methods to create instances:
    - WithoutBG.opensource() for local Open Source models
    - WithoutBG.api(api_key) for withoutBG Pro API
    """

    @staticmethod
    def opensource() -> "WithoutBGOpenSource":
        """Create instance using local Open Source models.

        Returns:
            WithoutBGOpenSource: Instance for local background removal

        Example:
            >>> model = WithoutBG.opensource()
            >>> result = model.remove_background("input.jpg")
        """
        return WithoutBGOpenSource()

    @staticmethod
    def api(api_key: str) -> "WithoutBGAPI":
        """Create instance using withoutBG Pro API.

        Args:
            api_key: API key for withoutBG Pro service

        Returns:
            WithoutBGAPI: Instance for cloud-based background removal

        Example:
            >>> model = WithoutBG.api(api_key="sk_...")
            >>> result = model.remove_background("input.jpg")
        """
        return WithoutBGAPI(api_key)

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Remove background from a single image.

        Args:
            input_image: Input image as file path, PIL Image, or bytes
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments passed to the model/API

        Returns:
            PIL Image with background removed
        """
        raise NotImplementedError("Subclass must implement remove_background()")

    def remove_background_batch(
        self,
        input_images: list[Union[str, Path, Image.Image, bytes]],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Remove background from multiple images.

        Args:
            input_images: List of input images
            output_dir: Directory to save results (optional)
            **kwargs: Additional arguments

        Returns:
            List of PIL Images with backgrounds removed
        """
        raise NotImplementedError("Subclass must implement remove_background_batch()")


class WithoutBGOpenSource(WithoutBG):
    """Local Open Source model implementation.

    Uses ONNX-based models running locally for background removal.
    Models are loaded once during initialization and reused for all inferences.
    """

    def __init__(
        self,
        depth_model_path: Optional[Union[str, Path]] = None,
        isnet_model_path: Optional[Union[str, Path]] = None,
        matting_model_path: Optional[Union[str, Path]] = None,
        refiner_model_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize with local Open Source models.

        Args:
            depth_model_path: Path to Depth Anything V2 ONNX model (optional)
            isnet_model_path: Path to ISNet segmentation ONNX model (optional)
            matting_model_path: Path to Matting ONNX model (optional)
            refiner_model_path: Path to Refiner ONNX model (optional)

        Note:
            Models are loaded once during initialization and cached in memory.
            If paths are not provided, models are downloaded from Hugging Face.
        """
        self.model = OpenSourceModel(
            depth_model_path=depth_model_path,
            isnet_model_path=isnet_model_path,
            matting_model_path=matting_model_path,
            refiner_model_path=refiner_model_path,
        )

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Remove background from image using local Open Source model.

        Args:
            input_image: Input image as file path, PIL Image, or bytes
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments (unused for Open Source model)

        Returns:
            PIL Image with background removed

        Example:
            >>> model = WithoutBG.opensource()
            >>> result = model.remove_background("input.jpg")
            >>> result.save("output.png")
        """
        try:
            return self.model.remove_background(
                input_image, progress_callback=progress_callback, **kwargs
            )
        except Exception as e:
            raise WithoutBGError(f"Background removal failed: {str(e)}") from e

    def remove_background_batch(
        self,
        input_images: list[Union[str, Path, Image.Image, bytes]],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Remove background from multiple images using local model.

        The model is loaded once and reused for all images, making this
        much more efficient than processing images separately.

        Args:
            input_images: List of input images
            output_dir: Directory to save results (optional)
            **kwargs: Additional arguments

        Returns:
            List of PIL Images with backgrounds removed

        Example:
            >>> model = WithoutBG.opensource()
            >>> results = model.remove_background_batch(["img1.jpg", "img2.jpg"])
        """
        results = []

        for i, input_image in enumerate(input_images):
            output_path = None
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)

                # Try to get original filename
                if isinstance(input_image, (str, Path)):
                    input_path = Path(input_image)
                    stem = input_path.stem
                    suffix = input_path.suffix or ".png"
                    output_filename = f"{stem}-withoutbg{suffix}"
                else:
                    # For PIL Images or bytes, use numbered fallback
                    output_filename = f"output_{i:04d}-withoutbg.png"

                output_path = output_dir_path / output_filename

            # Process image (reusing self.model for efficiency)
            result = self.remove_background(input_image, **kwargs)

            if output_path:
                result.save(output_path)
                # Note: Keep result in memory for return, don't close it yet

            results.append(result)

        return results


class WithoutBGAPI(WithoutBG):
    """withoutBG Pro API implementation.

    Uses cloud-based withoutBG Pro API for high-quality background removal.
    API client is initialized once and reused for all requests.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.withoutbg.com"):
        """Initialize withoutBG Pro API client.

        Args:
            api_key: API key for withoutBG Pro service
            base_url: Base URL for API endpoints (optional)

        Note:
            API client is initialized once and reused for all requests.
        """
        self.api_client = ProAPI(api_key=api_key, base_url=base_url)

    def remove_background(
        self,
        input_image: Union[str, Path, Image.Image, bytes],
        progress_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Remove background using withoutBG Pro API.

        Args:
            input_image: Input image as file path, PIL Image, or bytes
            progress_callback: Optional callback for progress updates
            **kwargs: Additional API parameters

        Returns:
            PIL Image with background removed

        Example:
            >>> model = WithoutBG.api(api_key="sk_...")
            >>> result = model.remove_background("input.jpg")
            >>> result.save("output.png")
        """
        try:
            return self.api_client.remove_background(
                input_image, progress_callback=progress_callback, **kwargs
            )
        except Exception as e:
            raise WithoutBGError(f"Background removal failed: {str(e)}") from e

    def remove_background_batch(
        self,
        input_images: list[Union[str, Path, Image.Image, bytes]],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Remove background from multiple images using API.

        The API client is reused for all images. Automatically adds a 3-second
        delay between requests to respect the 20 requests/minute rate limit.

        Args:
            input_images: List of input images
            output_dir: Directory to save results (optional)
            **kwargs: Additional arguments

        Returns:
            List of PIL Images with backgrounds removed

        Example:
            >>> model = WithoutBG.api(api_key="sk_...")
            >>> results = model.remove_background_batch(["img1.jpg", "img2.jpg"])
        """
        results = []

        for i, input_image in enumerate(input_images):
            output_path = None
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)

                # Try to get original filename
                if isinstance(input_image, (str, Path)):
                    input_path = Path(input_image)
                    stem = input_path.stem
                    suffix = input_path.suffix or ".png"
                    output_filename = f"{stem}-withoutbg{suffix}"
                else:
                    # For PIL Images or bytes, use numbered fallback
                    output_filename = f"output_{i:04d}-withoutbg.png"

                output_path = output_dir_path / output_filename

            # Process image (reusing self.api_client for efficiency)
            result = self.remove_background(input_image, **kwargs)

            if output_path:
                result.save(output_path)
                # Note: Keep result in memory for return, don't close it yet

            results.append(result)

            # Rate limit: 20 requests/minute = 3s per request
            # Skip delay after the last image
            if i < len(input_images) - 1:
                time.sleep(3)

        return results
