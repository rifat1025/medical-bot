from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# 1. Load PDF
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# 2. Split text into chunks
def split_texts(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    return texts

# 3. Create vector store with embeddings
def create_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="medical_db")
    vectorstore.persist()
    return vectorstore

# 4. Initialize chatbot with HuggingFacePipeline
def medical_chatbot(vectorstore):
    # Load a HF model (Flan-T5 small)
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chatbot

# 5. Run chatbot query
if __name__ == "__main__":
    docs = load_documents("Medical_book.pdf")
    chunks = split_texts(docs)
    vectorstore = create_vectorstore(chunks)
    bot = medical_chatbot(vectorstore)

    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = bot({"question": query, "chat_history": chat_history})
        print("Bot:", result["answer"])
        chat_history.append((query, result["answer"]))