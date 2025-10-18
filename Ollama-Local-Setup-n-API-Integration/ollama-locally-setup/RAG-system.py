import ollama
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("documents")


def load_documents(folder_path):
    """Load all text documents from a folder"""
    documents = []

    for file_path in Path(folder_path).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                "content": content,
                "source": file_path.name
            })

    print(f"✓ Loaded {len(documents)} documents")
    return documents


def chunk_text(text, chunk_size=500):
    """Split text into chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def build_knowledge_base(documents):
    """Add documents to vector database"""
    total_chunks = 0

    for doc in documents:
        chunks = chunk_text(doc["content"])

        for i, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()

            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"{doc['source']}_{i}"],
                metadatas=[{"source": doc['source']}]
            )

        total_chunks += len(chunks)

    print(f"✓ Added {len(documents)} documents ({total_chunks} chunks) to knowledge base")


def retrieve_context(question, n_results=3):
    """Find relevant document chunks for a question"""
    query_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results['documents'][0], results['metadatas'][0]


def query_without_rag(question):
    """Query Ollama WITHOUT RAG (for comparison)"""
    response = ollama.chat(model='llama2', messages=[
        {'role': 'user', 'content': question}
    ])

    return response['message']['content']


def query_with_rag(question):
    """Query WITH RAG"""
    # Get relevant context
    context_chunks, metadata = retrieve_context(question)
    context = "\n\n".join(context_chunks)

    # Show sources
    sources = list(set([meta['source'] for meta in metadata]))
    print(f"📚 Sources: {', '.join(sources)}")

    # Create enhanced prompt
    enhanced_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

    # Query Ollama
    response = ollama.chat(model='llama2', messages=[
        {'role': 'user', 'content': enhanced_prompt}
    ])

    return response['message']['content']


# Setup
print("\n=== Setting up RAG System ===\n")
docs = load_documents("documents/")
build_knowledge_base(docs)

# Test mode selection
print("\n=== RAG Chatbot Ready ===")
print("Commands:")
print("  'rag on'  - Use RAG (default)")
print("  'rag off' - No RAG (for comparison)")
print("  'compare <question>' - Show both responses")
print("  'quit' or 'exit' - Exit\n")

use_rag = True

while True:
    question = input("You: ")

    if question.lower() in ['quit', 'exit']:
        break

    if question.lower() == 'rag on':
        use_rag = True
        print("✓ RAG enabled\n")
        continue

    if question.lower() == 'rag off':
        use_rag = False
        print("✓ RAG disabled\n")
        continue

    if question.lower().startswith('compare '):
        actual_question = question[8:]

        print("\n--- WITHOUT RAG ---")
        response_no_rag = query_without_rag(actual_question)
        print(f"Bot: {response_no_rag}\n")

        print("--- WITH RAG ---")
        response_with_rag = query_with_rag(actual_question)
        print(f"Bot: {response_with_rag}\n")
        continue

    # Normal query
    if use_rag:
        response = query_with_rag(question)
    else:
        response = query_without_rag(question)

    print(f"\nBot: {response}\n")