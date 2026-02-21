
# Medical Chat Bot

A Django-based medical document chatbot using LangChain and Hugging Face models.

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Medical Document
Place your PDF file in the project root:
```bash
Medical_book.pdf
```

## Build

### 1. Initialize Database
```bash
python manage.py migrate
```

### 2. Create Vector Store
```bash
python manage.py shell
>>> from medical_chat.chatbot import get_vectorstore
>>> get_vectorstore()
>>> exit()
```

## Run

### Start Development Server
```bash
python manage.py runserver
```

Access at `http://localhost:8000`

## Project Structure
- `medical_chat/chatbot.py` - LangChain RAG pipeline
- `medical_chat/views.py` - Django views
- `medical_chat/models.py` - Chat message storage
- `bot/requirements.txt` - Dependencies
