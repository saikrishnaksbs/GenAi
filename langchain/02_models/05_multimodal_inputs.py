"""
MULTIMODAL INPUTS
===================
Modern chat models can accept more than text: images, PDFs, and audio can
be embedded directly inside a HumanMessage's `content` list. Instead of a
plain string, `content` becomes a list of typed content blocks, each with
a `type` key ("text", "image_url", "file", ...).

This is the standard cross-provider format LangChain normalizes to, so the
same message structure works (with minor provider quirks) across
ChatOpenAI, ChatAnthropic, and others.
"""

import os
import base64
import urllib.request

from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama

# Helper to ensure dummy files exist for the demo
def ensure_dummy_files():
    if not os.path.exists("diagram.png"):
        # Tiny 1x1 transparent PNG
        png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'
        with open("diagram.png", "wb") as f:
            f.write(png_bytes)
        print("Created dummy diagram.png for the demo.")
        
    if not os.path.exists("report.pdf"):
        # Tiny dummy PDF bytes
        pdf_bytes = b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R/Resources<<>>/Contents 4 0 R>>endobj\n4 0 obj<</Length 12>>stream\nHello World\nendstream\nendobj\ntrailer<</Size 5/Root 1 0 R>>\n%%EOF'
        with open("report.pdf", "wb") as f:
            f.write(pdf_bytes)
        print("Created dummy report.pdf for the demo.")

ensure_dummy_files()

# Qwen2.5:1.5b is a text-only model. To actually run multimodal inputs,
# you should run: ollama pull llava
# and change the model name below to "llava".
model = ChatOllama(model="qwen2.5:1.5b")

# --- Image via URL -------------------------------------------------------
# Note: Ollama requires local or base64 encoded images. We fetch the image and
# base64-encode it so it works cleanly.
try:
    image_url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langgraph_overview.png"
    with urllib.request.urlopen(image_url, timeout=5) as conn:
        url_bytes = conn.read()
    url_b64 = base64.b64encode(url_bytes).decode("utf-8")
    image_url_val = f"data:image/png;base64,{url_b64}"
except Exception:
    # Fallback to local dummy base64 if network is down
    image_url_val = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

message_with_url_image = HumanMessage(
    content=[
        {"type": "text", "text": "What's unusual about this image?"},
        {
            "type": "image_url",
            "image_url": {"url": image_url_val},
        },
    ]
)

print("\n--- Testing Multimodal URL Image ---")
try:
    response = model.invoke([message_with_url_image])
    print(response.content)
except Exception as e:
    print(f"Skipped/Failed: {e}")
    print("Note: Ollama requires a multimodal model (like 'llava') to process images.")

# --- Image via base64-encoded bytes --------------------------------------
with open("diagram.png", "rb") as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

message_with_inline_image = HumanMessage(
    content=[
        {"type": "text", "text": "Summarize this diagram."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        },
    ]
)

print("\n--- Testing Multimodal Inline Image ---")
try:
    response = model.invoke([message_with_inline_image])
    print(response.content)
except Exception as e:
    print(f"Skipped/Failed: {e}")

# --- File input (e.g. PDF) ------------------------------------------------
# Some providers (like Anthropic) accept raw file blocks for PDFs.
with open("report.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

message_with_pdf = HumanMessage(
    content=[
        {"type": "text", "text": "Summarize the key findings in this report."},
        {
            "type": "file",
            "source_type": "base64",
            "mime_type": "application/pdf",
            "data": pdf_b64,
        },
    ]
)

print("\n--- Testing Multimodal PDF file ---")
try:
    response = model.invoke([message_with_pdf])
    print(response.content)
except Exception as e:
    print(f"Skipped/Failed: {e}")

