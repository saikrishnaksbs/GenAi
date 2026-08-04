"""
MULTI-MODAL RAG (IMAGES & TABLES INSIDE DOCUMENTS)
==================================================
Many production documents (e.g. financial charts, product brochures) contain vital 
insights locked in images, diagrams, and tables that plain text parsers ignore.

This script demonstrates a Multi-modal RAG pipeline:
1. Parse text and extract images/tables.
2. Use a multimodal LLM (gpt-4o) to generate text summaries for each image/table.
3. Index both the original text chunks and the image summaries in a single Vector Store,
   linking the summaries to the original image files.
4. On query retrieval, fetch the closest text or image summaries.
5. Feed both retrieved text context and original images to the model for final generation.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="qwen2.5:1.5b", temperature=0)

# --------------------------------------------------------------------------
# Step 1 & 2: Image Summary Extraction Setup
# --------------------------------------------------------------------------
# Suppose we have parsed a PDF and extracted one text page and one image file.
parsed_text_chunk = "Company revenue grew by 15% in Q3, reaching a record high of $45M."
mock_image_path = "financial_chart_q3.png"

# We write a prompt to summarize the image content for text-based retrieval
async def generate_image_summary(image_url_or_base64: str) -> str:
    """Invokes multimodal model to describe a chart image."""
    # In a real environment, you'd pass a base64 string or image URI
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this financial chart or image in detail. Extract any numbers, dates, or trends shown."},
            {
                "type": "image_url",
                "image_url": {"url": "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langgraph_overview.png"}, # sample image URL
            },
        ]
    )
    # response = await model.ainvoke([message])
    # return response.content
    return "Chart summary: LangGraph orchestrates multi-agent systems via state graphs."


# --------------------------------------------------------------------------
# Step 3: Indexing summaries in VectorStore
# --------------------------------------------------------------------------
async def index_multimodal_context():
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    
    # Get the summary of the image
    img_summary = await generate_image_summary(mock_image_path)
    
    # We construct Document objects. For the image summary, we retain the 
    # original image file path in the metadata.
    documents = [
        Document(
            page_content=parsed_text_chunk, 
            metadata={"source": "report_q3.pdf", "type": "text"}
        ),
        Document(
            page_content=img_summary, 
            metadata={"source": "report_q3.pdf", "type": "image", "image_path": mock_image_path}
        )
    ]
    
    db = FAISS.from_documents(documents, embeddings)
    print("Indexed text chunk and image summary in FAISS.")
    return db


# --------------------------------------------------------------------------
# Step 4 & 5: Retrieval and Multimodal Generation
# --------------------------------------------------------------------------
async def query_multimodal_rag(db, user_query: str):
    # 1. Search vector database
    retriever = db.as_retriever(search_kwargs={"k": 2})
    matched_docs = retriever.invoke(user_query)
    
    print(f"\nQuery: '{user_query}'")
    print(f"Retrieved {len(matched_docs)} documents.")
    
    text_context = []
    images_to_feed = []
    
    for doc in matched_docs:
        print(f"  -> Type: {doc.metadata['type']} (Source: {doc.metadata['source']})")
        if doc.metadata["type"] == "text":
            text_context.append(doc.page_content)
        elif doc.metadata["type"] == "image":
            text_context.append(f"[Image Summary]: {doc.page_content}")
            # Track the original file to pass to LLM
            images_to_feed.append(doc.metadata["image_path"])
            
    # 2. Formulate final Multimodal prompt
    # In a real environment, you load images from disk, convert to base64, 
    # and append to the message block payload.
    prompt_content = [
        {"type": "text", "text": f"Answer the user query using the retrieved context and matching images.\n\nContext:\n" + "\n".join(text_context)},
        {"type": "text", "text": f"User Query: {user_query}"}
    ]
    
    # If the retrieved context includes images, we attach them base64 encoded
    for img_path in images_to_feed:
        print(f"Attaching image file '{img_path}' to the generation payload.")
        # prompt_content.append({"type": "image_url", "image_url": ...})

    # response = await model.ainvoke([HumanMessage(content=prompt_content)])
    # print(f"Response: {response.content}")
    print("Successfully structured multimodal RAG query payload.")


if __name__ == "__main__":
    async def main():
        db = await index_multimodal_context()
        await query_multimodal_rag(db, "Summarize how agents are coordinated in the graph chart.")

    asyncio.run(main())
