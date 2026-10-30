from pathlib import Path
from typing import Annotated

import dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.model.model_gen import ChatService
from src.services import DocumentService, PromptService, VectorStoreService

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

dotenv.load_dotenv()

vector_store_svc = VectorStoreService()
document_svc = DocumentService(upload_dir=UPLOAD_DIR)
prompt_svc = PromptService()
model_api = ChatService()

app = FastAPI()

STATIC_DIR = Path("src/static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    # Serve the chatbot UI
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/upload")
async def upload_file(file: Annotated[UploadFile, File()]):

    # Save the uploaded file
    file_path = document_svc.save_file(file)

    # Load the text from the saved file
    documents = document_svc.load_file(file_path)

    # Split the documents into chunks
    chunks = document_svc.split_documents(documents, source=str(file_path))

    vector_store_svc.add_documents(chunks)

    return {"filename": file.filename, "message": "File uploaded successfully"}


@app.post("/ask")
async def ask_question(question: str):
    hits = vector_store_svc.search(question, k=5)

    context = document_svc.format_context(hits)

    prompt, system = prompt_svc.get_prompt()
    prompt = prompt.format(context=context, question=question)

    response = model_api.generate_content(system, prompt)
    answer = response.choices[0].message.content

    return {"question": question, "answer": answer}
