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

import base64

from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b")

# --- Image via URL -------------------------------------------------------
message_with_url_image = HumanMessage(
    content=[
        {"type": "text", "text": "What's unusual about this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"},
        },
    ]
)

response = model.invoke([message_with_url_image])
print(response.content)
# -> "The image shows a cat sitting inside a fishbowl, which is unusual..."

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

response = model.invoke([message_with_inline_image])
print(response.content)
# -> "The diagram illustrates a three-tier architecture with..."

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

response = model.invoke([message_with_pdf])
print(response.content)
# -> "The report finds that quarterly revenue grew 12% year over year..."
