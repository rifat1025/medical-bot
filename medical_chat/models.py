# chat/models.py
from django.db import models

class ChatMessage(models.Model):
    user_question = models.TextField()
    bot_answer = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user_question[:50]}..."