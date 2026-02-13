# Multimodal Messages

Blackgeorge supports multimodal messages allowing you to send images, videos, audio, and documents to vision-capable models.

## Overview

Multimodal messages use the litellm/OpenAI format where `content` can be either a string or a list of content objects.

## Sending Images

### Using URLs

```python
from blackgeorge import Desk, Job, Worker

desk = Desk(model="openrouter/google/gemini-3-flash-preview")
worker = Worker(name="VisionAnalyst")

job = Job(input=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
])

report = desk.run(worker, job)
print(report.content)
```

### Using Local Files

```python
from blackgeorge import Desk, Job, Worker, encode_file

image_data = encode_file("./my_photo.jpg")

job = Job(input=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image_url", "image_url": {"url": image_data}}
])

report = desk.run(worker, job)
```

### Multiple Images

```python
job = Job(input=[
    {"type": "text", "text": "Compare these two images"},
    {"type": "image_url", "image_url": {"url": "https://example.com/before.jpg"}},
    {"type": "image_url", "image_url": {"url": "https://example.com/after.jpg"}}
])
```

## Sending Videos

```python
job = Job(input=[
    {"type": "text", "text": "Summarize this video"},
    {"type": "video_url", "video_url": {"url": "https://youtube.com/watch?v=..."}}
])

# For local video files
video_data = encode_file("./demo.mp4")
job = Job(input=[
    {"type": "text", "text": "What happens in this video?"},
    {"type": "video_url", "video_url": {"url": video_data}}
])
```

## Sending Audio

```python
audio_data = encode_file("./recording.mp3")

job = Job(input=[
    {"type": "text", "text": "Transcribe this audio"},
    {"type": "file", "file": {"file_data": audio_data}}
])
```

## Document Understanding (PDF, DOCX, etc.)

```python
pdf_data = encode_file("./contract.pdf")

job = Job(input=[
    {"type": "text", "text": "Extract key terms from this contract"},
    {"type": "file", "file": {
        "file_data": pdf_data,
        "filename": "contract.pdf"
    }}
])
```

Supported document types:
- PDF (`.pdf`)
- Word (`.doc`, `.docx`)
- Excel (`.xls`, `.xlsx`)
- CSV (`.csv`)
- HTML (`.html`)
- Markdown (`.md`)
- Text (`.txt`)

## Generating Images

Blackgeorge includes tools for generating images with OpenRouter, Gemini, and OpenAI-compatible
providers:

```python
from blackgeorge import Desk, Job, Worker, generate_image

desk = Desk(model="openrouter/google/gemini-3-flash-preview")
worker = Worker(name="Assistant", tools=[generate_image])

job = Job(input="Please generate an image of a sunset over mountains")
report = desk.run(worker, job)
```

The `generate_image` tool accepts:
- `prompt` (required): Text description of the image
- `model` (optional): Model to use (default:
  `openrouter/google/gemini-3-pro-image-preview`)
- `size` (optional): Image size (default: `1024x1024`)
- `quality` (optional): Image quality (default: `standard`)

Returns a dict with:
- `url`: URL of the generated image
- `b64_json`: Base64-encoded image data when returned by the provider
- `revised_prompt`: The model's revised prompt when supported

For models that generate images through chat completions, Blackgeorge automatically falls back to
`completion(..., modalities=["image", "text"])` if the image endpoint returns no image data.

## The `encode_file()` Utility

`encode_file(file_path, mime_type=None)` converts local files to base64 data URLs:

```python
from blackgeorge import encode_file

# Auto-detect MIME type from extension
image_data = encode_file("photo.jpg")  # data:image/jpeg;base64,...
pdf_data = encode_file("doc.pdf")      # data:application/pdf;base64,...

# Explicit MIME type
data = encode_file("file.bin", mime_type="application/octet-stream")
```

## Supported Models

Multimodal input support varies by model:

| Feature | Models |
|---------|--------|
| Images | `openrouter/google/gemini-3-flash-preview`, `openrouter/google/gemini-3-pro-preview` |
| Videos | `openrouter/google/gemini-3-flash-preview`, `openrouter/google/gemini-3-pro-preview` |
| Audio | `openrouter/google/gemini-3-flash-preview`, `openrouter/google/gemini-3-pro-preview` |
| Documents | `openrouter/google/gemini-3-flash-preview`, `openrouter/google/gemini-3-pro-preview` |
| Image Generation | `openrouter/google/gemini-3-pro-image-preview` |

## Examples

### Vision Analysis with Gemini 3 Flash Preview

```python
from blackgeorge import Desk, Job, Worker, encode_file

desk = Desk(model="openrouter/google/gemini-3-flash-preview")
worker = Worker(name="VisionBot", instructions="You analyze images in detail")

photo = encode_file("./product.jpg")
job = Job(input=[
    {"type": "text", "text": "Describe this product in detail for a catalog"},
    {"type": "image_url", "image_url": {"url": photo}}
])

report = desk.run(worker, job)
print(report.content)
```

### Document Q&A with Gemini 3 Pro Preview

```python
from blackgeorge import Desk, Job, Worker, encode_file

desk = Desk(model="openrouter/google/gemini-3-pro-preview")
worker = Worker(name="DocumentAnalyst")

pdf = encode_file("./research_paper.pdf")
job = Job(input=[
    {"type": "text", "text": "What are the main findings of this research?"},
    {"type": "file", "file": {"file_data": pdf, "filename": "research_paper.pdf"}}
])

report = desk.run(worker, job)
```

## Best Practices

1. **File Size**: Large files (>10MB) may cause timeouts or errors
2. **Model Limits**: Check model documentation for maximum image/video duration limits
3. **Cost**: Multimodal requests typically cost more than text-only
4. **URLs vs Base64**: URLs are more efficient for remote files; use base64 for local files
5. **Error Handling**: Always validate that your chosen model supports the media type
