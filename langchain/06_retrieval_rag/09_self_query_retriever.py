"""
SELF QUERY RETRIEVER
=====================
Sometimes a query mixes semantic intent with structured constraints, e.g.
"cheap sci-fi movies from the 1990s". `SelfQueryRetriever` uses an LLM to
split that into (1) a semantic search string and (2) a structured
metadata filter, by inspecting a description of the document schema you
provide. The filter is then applied at the vector store level alongside
the similarity search.
"""

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

docs = [
    Document(
        page_content="A group of astronauts travel through a wormhole in search of a new home for humanity.",
        metadata={"genre": "sci-fi", "year": 2014, "rating": 8.6},
    ),
    Document(
        page_content="A computer hacker learns the true nature of his reality.",
        metadata={"genre": "sci-fi", "year": 1999, "rating": 8.7},
    ),
    Document(
        page_content="A chef starts his own restaurant and struggles to keep it alive.",
        metadata={"genre": "drama", "year": 2007, "rating": 7.5},
    ),
]

embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
vectorstore = Chroma.from_documents(docs, embeddings)

# Describe each metadata field so the LLM knows what it can filter on and how.
metadata_field_info = [
    AttributeInfo(name="genre", description="The genre of the movie", type="string"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="rating", description="A 1-10 rating of the movie", type="float"),
]
document_content_description = "Brief summary of a movie"

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    enable_limit=True,  # allows the LLM to also infer a result-count limit, e.g. "top 2"
)

# The LLM parses this into something like:
#   query="astronauts space travel", filter=Comparison(genre == "sci-fi" AND year > 2000)
results = self_query_retriever.invoke("sci-fi movies released after 2000")

print([d.metadata["year"] for d in results])
# -> [2014]

# A pure filter query with no real semantic component still works.
top_rated = self_query_retriever.invoke("movies rated above 8.5")
print([d.metadata["rating"] for d in top_rated])
# -> [8.6, 8.7]
