# chat/chatbot.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate

VECTORSTORE_PATH = "medical_chat/medical_db"
PDF_PATH = "Medical_book.pdf"

# 1. Load PDF documents
def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    return documents

# 2. Split documents into chunks
def split_texts():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = splitter.split_documents(docs)
    return texts

# 3. Create or load vectorstore
def get_vectorstore():
    texts = split_texts()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=VECTORSTORE_PATH)
    vectorstore.persist()
    return vectorstore

# 4. Initialize chatbot with system prompt
def get_chatbot():
    vectorstore = get_vectorstore()

    # Load Hugging Face model
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # System prompt to guide the bot
    system_prompt = """You are a medical document assistant.

Rules:
1. Use ONLY the provided context.
2. If the answer is not in the context, say: "The document does not contain this information."
3. Do NOT provide diagnosis, treatment plans, or personal medical advice.
4. Do NOT speculate.
"""

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{system_prompt}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\nAnswer:"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    return chatbot

# 5. Initialize the bot once
bot = get_chatbot()