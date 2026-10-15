from langchain_huggingface import HuggingFaceEmbeddings


def embed_documents(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    result = embeddings.embed_documents(documents)
    return result
