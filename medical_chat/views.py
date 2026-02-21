# chat/views.py
from django.shortcuts import render
from .chatbot import bot
from .models import ChatMessage 

def index(request):
    answer = None

    if request.method == "POST":
        question = request.POST.get("question")
        if question:
            # Optional: load previous chat history from DB
            chat_history = [(chat.user_question, chat.bot_answer) for chat in ChatMessage.objects.all().order_by('timestamp')] if ChatMessage.objects.exists() else []

            # Ask the bot
            response = bot({"question": question, "chat_history": chat_history})
            answer = response["answer"]

            # Save to DB
            ChatMessage.objects.create(user_question=question, bot_answer=answer)

    # Load all chat messages for display
    all_chats = ChatMessage.objects.all().order_by('timestamp') if ChatMessage.objects.exists() else []
    chat_history = [(chat.user_question, chat.bot_answer) for chat in all_chats]

    return render(request, "medical_chat/index.html", {"chat_history": chat_history, "answer": answer})
