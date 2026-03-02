# withoutbg

**Background removal that runs locally or through the API. Pick whichever fits your constraints.**

[![PyPI](https://img.shields.io/pypi/v/withoutbg.svg)](https://pypi.org/project/withoutbg/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Next_Gen_Model-Coming_Feb_2026-blue?style=for-the-badge&logo=github)](https://github.com/withoutbg/withoutbg/stargazers)

Two modes: run the open source Focus model locally (free, private, works offline), or call the withoutBG Pro API (better quality, no GPU needed, pay per image). The code is the same either way; just swap the initializer.


## Try It

![Python Package Intro](/images/python-package-intro.png)

```bash
pip install withoutbg
```

```python
from withoutbg import WithoutBG

model = WithoutBG.opensource()
model.remove_background("your-photo.jpg").save("result.png")
```

## Local vs. API: The Real Tradeoff

The local model loads ~320MB of weights into ~2GB of RAM. You pay that cost once per process, then process images for free indefinitely. The API skips all of that: send an image, get a result in 1-3 seconds, pay per call.

If you're building a product, the API is the right default. You don't want to manage 2GB of model weights in a production service, and the quality is better. The local model makes sense when you need offline or private processing, or when you're running a large batch job and don't want to pay per image.

```
Need offline or private processing?     → Local model
Processing a large batch?               → Local model (pay once, amortize over all images)
Building a product?                     → withoutBG Pro (better quality, zero infra overhead)
Occasional use, no setup tolerance?     → withoutBG Pro
```

**[Compare Focus vs Pro →](https://withoutbg.com/resources/compare/focus-vs-pro?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**

## Quick Start

### Docker (Web Interface)

**[View Complete Dockerized Web App Documentation →](https://withoutbg.com/documentation/integrations/dockerized-web-app?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**

![Web Application in Docker](/images/dockerized-app.png)
```bash
docker run -p 80:80 withoutbg/app:latest
open http://localhost
```

Runs on both amd64 and arm64.

### Python SDK

**[View Complete Python SDK Documentation →](https://withoutbg.com/documentation/integrations/python-sdk?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**

```bash
uv add withoutbg
# or: pip install withoutbg
```

**Local model:**
```python
from withoutbg import WithoutBG

model = WithoutBG.opensource()
result = model.remove_background("input.jpg")  # Returns PIL Image (RGBA)
result.save("output.png")

result.show()
result.resize((500, 500))
result.save("output.webp", quality=95)
```

**withoutBG Pro:**

**[See Pro API Results →](https://withoutbg.com/resources/background-removal-results/model-pro-api?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**

```python
from withoutbg import WithoutBG

model = WithoutBG.api(api_key="sk_your_key")
result = model.remove_background("input.jpg")
result.save("output.png")

# Prefer the environment variable so the key stays out of your code
# export WITHOUTBG_API_KEY=sk_your_key
```

**CLI:**
```bash
withoutbg photo.jpg

withoutbg ~/Photos/vacation/ --batch --output-dir ~/Photos/vacation-no-bg/

# Flatten to JPEG with white fill (for printing or upload)
withoutbg portrait.jpg --format jpg --quality 95

export WITHOUTBG_API_KEY=sk_your_key
withoutbg wedding-photo.jpg --use-api

withoutbg photo.jpg --verbose
```

> Don't have `uv`? It's a faster drop-in for pip. Get it at [astral.sh/uv](https://astral.sh/uv).

### Example Outputs from the Open Source Model

**[See More Focus Model Results →](https://withoutbg.com/resources/background-removal-results/model-focus-open-source?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**

![Example 1](/sample-results/open-source-focus/example1.png)
![Example 5](/sample-results/open-source-focus/example5.png)
![Example 3](/sample-results/open-source-focus/example3.png)
![Example 6](/sample-results/open-source-focus/example6.png)
![Example 4](/sample-results/open-source-focus/example4.png)


## Repository Structure

Monorepo with three layers: packages (reusable libraries), apps (end-user deployments), and integrations (plugin targets):

```
withoutbg/
├── packages/          # Reusable packages
│   └── python/        # Core Python SDK (published to PyPI)
│
├── apps/              # End-user applications
│   └── web/           # Web application (React + FastAPI)
│
├── integrations/      # Third-party tool integrations
│   └── (future: GIMP, Photoshop, Figma plugins)
│
├── models/            # Shared ML model files
│   └── checkpoints/   # ONNX model files
│
├── docs/              # Documentation
└── scripts/           # Development scripts
```

### Components

#### [Python SDK](packages/python/)
Core library. Published to PyPI as `withoutbg`.

- **Install**: `uv add withoutbg` (or `pip install withoutbg`)
- **Exposes**: Python API + CLI
- **Models**: Focus v1.0.0 (local), withoutBG Pro (API)

#### [Web Application](apps/web/)
Web interface with drag-and-drop, batch processing, and live preview.

- **Stack**: React 18 + FastAPI + Nginx
- **Deploy**: Docker Compose

#### Integrations (Coming Soon)
- GIMP plugin
- Photoshop extension
- Figma plugin
- Blender addon

## Usage Notes

### What gets returned

All methods return a PIL `Image` in RGBA mode, a standard Python image object with an alpha channel carrying the mask:

```python
from withoutbg import WithoutBG

model = WithoutBG.opensource()
result = model.remove_background("photo.jpg")  # PIL Image, RGBA

result.save("output.png")    # PNG preserves the alpha channel
result.save("output.webp")   # WebP also supports alpha
result.save("output.jpg", quality=95)  # JPEG drops alpha; you get a flat image
```

### JPEG silently drops transparency

JPEG has no alpha channel. Saving to `.jpg` will discard the mask without an error:

```python
# This works but you lose the cutout:
result.save("output.jpg")

# Use PNG or WebP to keep it:
result.save("output.png")
result.save("output.webp")
```

### First-run model download

On first call, `WithoutBG.opensource()` downloads ~320MB of ONNX weights from HuggingFace and caches them locally. This takes 5-10 seconds depending on your connection. Every subsequent run loads from cache and processes in 2-5 seconds per image.

```python
# First call: downloads 320MB, takes 5-10s
model = WithoutBG.opensource()

# Every call after that: loads from cache, fast
result = model.remove_background("photo.jpg")
```

### Batch processing

The model weights stay in RAM as long as the `model` object is alive. If you create a new `WithoutBG` instance for every image, you're loading and unloading 2GB of RAM each time. Don't do that:

```python
from withoutbg import WithoutBG

model = WithoutBG.opensource()

images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
results = model.remove_background_batch(images, output_dir="results/")
```

### Output naming

- Single file: `photo.jpg` → `photo-withoutbg.png`
- Batch: `photo1-withoutbg.png`, `photo2-withoutbg.png`, etc.
- Override: `--output` (single) or `--output-dir` (batch)

### Progress callback

```python
def show_progress(progress):
    print(f"Processing: {progress * 100:.0f}%")

model = WithoutBG.opensource()
result = model.remove_background("photo.jpg", progress_callback=show_progress)
```

### Error handling

```python
from withoutbg import WithoutBG, APIError, WithoutBGError

try:
    model = WithoutBG.api(api_key="sk_your_key")
    result = model.remove_background("photo.jpg")
    result.save("output.png")
except APIError as e:
    print(f"API error: {e}")
except WithoutBGError as e:
    print(f"Processing error: {e}")
```

## Performance

| Metric | Local (CPU) | withoutBG Pro |
|--------|-------------|---------------|
| **First Run** | 5-10s (~320MB download) | 1-3s |
| **Per Image** | 2-5s | 1-3s |
| **Memory** | ~2GB RAM | None |
| **Disk Space** | 320MB (one-time) | None |
| **Setup** | One-time download | API key only |
| **Cost** | Free forever | Pay per use |

Model breakdown (cached after first download):

- ISNet segmentation: 177 MB
- Depth Anything V2: 99 MB
- Focus Matting: 27 MB
- Focus Refiner: 15 MB
- Total: ~320 MB

For batch jobs, keep the model object alive across all images. Reinitializing for each image reloads the weights every time, which is 10-100x slower than reusing a single instance.

## Troubleshooting

**Model download fails:**
- Models are pulled from HuggingFace on first run (~320MB). Check your connection.
- To use locally cached or custom model files, see [Configuration](packages/python/README.md#configuration).

**Out of memory:**
- The local model uses ~2GB of RAM. Either reduce your batch size or switch to the API, which offloads inference entirely.

**Import error or "module not found":**
```bash
which python          # confirm you're in the right environment
pip list | grep withoutbg

source venv/bin/activate
pip install withoutbg
```

**API key rejected:**
- Get your key at [withoutbg.com](https://withoutbg.com).
- Set it as an environment variable: `export WITHOUTBG_API_KEY=sk_your_key`

**Slow on first run:**
- Expected. The ~320MB weights are downloading. Subsequent runs use the local cache and take 2-5s per image.

## Documentation

- **[Python SDK Docs](packages/python/README.md)**: API reference and examples
- **[Python SDK Documentation](https://withoutbg.com/documentation/integrations/python-sdk?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**: Online documentation
- **[Web App Docs](apps/web/README.md)**: Deployment and development guide
- **[Dockerized Web App Documentation](https://withoutbg.com/documentation/integrations/dockerized-web-app?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**: Online documentation
- **[withoutBG Pro API Results](https://withoutbg.com/resources/background-removal-results/model-pro-api?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**: Example outputs
- **[Focus Model Results](https://withoutbg.com/resources/background-removal-results/model-focus-open-source?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**: Example outputs
- **[Compare Focus vs Pro](https://withoutbg.com/resources/compare/focus-vs-pro?utm_source=github&utm_medium=withoutbg-readme&utm_campaign=main-readme)**: Model comparison

## Development

### Python Package

```bash
cd packages/python

uv sync --extra dev
# or: pip install -e ".[dev]"

pytest

black src/ tests/
ruff check src/ tests/
```

### Web Application

```bash
docker-compose -f apps/web/docker-compose.yml up

# Or run the components separately:
cd apps/web/backend
uv sync
uvicorn app.main:app --reload

cd apps/web/frontend
npm install
npm run dev
```

## Focus v1.0.0

The current open source model. Key improvements over prior versions:

- Better edge detail, particularly around hair and fur (the hardest case for matting models)
- More consistent generalization across image types
- Pipeline: ISNet segmentation → Depth Anything V2 guidance → Focus Matting → Focus Refiner

See [sample-results/](sample-results/) for visual comparisons.

## License

Apache License 2.0. See [LICENSE](LICENSE)

### Third-Party Components
- **Depth Anything V2**: Apache 2.0 License (vits model only)
- **ISNet**: Apache 2.0 License

### Acknowledgements
- **[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)**: MIT License (used to train the matting and refiner models)

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for complete attribution.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md), then open a PR with tests. Small, focused changes are easiest to review.

## Support

- **Bugs**: [GitHub Issues](https://github.com/withoutbg/withoutbg/issues)
- **Discussion**: [GitHub Discussions](https://github.com/withoutbg/withoutbg/discussions)
- **Commercial**: [contact@withoutbg.com](mailto:contact@withoutbg.com)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=withoutbg/withoutbg&type=Date)](https://star-history.com/#withoutbg/withoutbg&Date)
