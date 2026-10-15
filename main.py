from io import BytesIO
from pathlib import Path

from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pypdf import PdfReader

from ChatModels.groq import get_llm_chat_result
from EmbeddedModels.local_embed_model import embed_documents
from Indexes.splitter import splitter_user_input_file
from Prompts.resarcher_prompt import get_researcher_prompt
from VectorDb.connect import QdrantConnect

app = FastAPI()


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/query")
async def echo_query(query: str = Form(...)) -> dict:
    # vectorise it
    # call embedding model
    # embedded_docs = embed_documents([query])
    qc = QdrantConnect()
    res = qc.query_db(collection="hr_policy", query=query)
    prompt = get_researcher_prompt(prompt=query, context=res.payload.get("text"))
    res = get_llm_chat_result(prompt)
    return {"query": res.content}


@app.post("/upload")
async def upload_document(
    query: str = Form(...),
    document: UploadFile = File(...),
) -> dict[str, str]:
    extension = Path(document.filename or "").suffix.lower()
    contents = await document.read()

    if extension == ".pdf":
        text = "\n".join(
            page.extract_text() or "" for page in PdfReader(BytesIO(contents)).pages
        )
    elif extension == ".docx":
        text = "\n".join(
            paragraph.text for paragraph in Document(BytesIO(contents)).paragraphs
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF or DOCX document.",
        )

    # print(f"Document text:\n{text}", flush=True)
    # print(f"User query:\n{query}", flush=True)
    splitted_text = splitter_user_input_file(text)
    # call embedding model
    # embedded_docs = embed_documents(splitted_text)
    qc = QdrantConnect()
    qc.save_db("hr_policy", splitted_text)
    return {"message": "Document text and query printed to the API console."}
