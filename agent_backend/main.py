import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

app = FastAPI(title="Enterprise Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHROMA_DIR = Path(__file__).parent.parent / "data_ingestion" / "chroma_db"
COLLECTION_NAME = "enterprise_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# State
chroma_client = None
collection = None
embedder = None

class ChatRequest(BaseModel):
    message: str

class Source(BaseModel):
    text: str
    source_file: str
    section_title: str
    score: float

class ChatResponse(BaseModel):
    reply: str
    sources: list[Source]

@app.on_event("startup")
def startup_event():
    global chroma_client, collection, embedder
    print(f"Loading ChromaDB from {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    print(f"Loading Embedding Model {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global collection, embedder
    
    if not collection or not embedder:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # 1. Embed the query
    query_vector = embedder.encode([request.message])[0].tolist()

    # 2. Search ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    sources = []
    context_texts = []
    
    if results and results["documents"] and len(results["documents"][0]) > 0:
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            score = round(1 - dist, 4)
            sources.append(Source(
                text=doc,
                source_file=meta.get("source_file", "Unknown"),
                section_title=meta.get("section_title", "Unknown"),
                score=score
            ))
            context_texts.append(f"Document: {meta.get('source_file')} - {meta.get('section_title')}\nContent: {doc}")

    # 3. Generate Answer (using Gemini if key is provided, else mock)
    context_str = "\n\n".join(context_texts)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""You are a helpful enterprise AI assistant. Answer the user's query using ONLY the provided context. If the context does not contain the answer, politely say you don't know based on the provided documents.
        
Context:
{context_str}

User Query: {request.message}
"""
        try:
            response = model.generate_content(prompt)
            reply = response.text
        except Exception as e:
            reply = f"Error generating response from LLM: {str(e)}"
    else:
        # Mock reply if no API key
        if context_texts:
            reply = "I found some relevant information in the documents. (Set GEMINI_API_KEY to generate a synthesized response.)\n\nBased on what I found, please see the sources below."
        else:
            reply = "I couldn't find any relevant information in the documents."

    return ChatResponse(reply=reply, sources=sources)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
