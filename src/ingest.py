from src.services import VectorStoreService, DocumentService, PromptService
from src.model.model_gen import ChatService


vector_store_svc = VectorStoreService()
document_svc = DocumentService(upload_dir='uploads')
prompt_svc = PromptService()
model_api = ChatService()

def ingest(file_path):
    # Load the text from the saved filea
    documents = document_svc.load_file(file_path)

    # Split the documents into chunks
    chunks = document_svc.split_documents(documents, source=str(file_path))

    vector_store_svc.add_documents(chunks)

def main():
    ingest("TechCorp_Official_Employee_Handbook.pdf")

if __name__ == "__main__":
    main()