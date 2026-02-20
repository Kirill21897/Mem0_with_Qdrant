import os
import time
from dotenv import load_dotenv
from mem0 import Memory
from config import MEM0_CONFIG

# Загрузка переменных окружения
load_dotenv(override=True)

# Инициализация клиента
client = Memory.from_config(MEM0_CONFIG)

# --- Логика из ноутбука .ipynb ---

user_id = "customer-001"
message_from_user_1 = "My order #1234 was for a 'Nova 2000', but it arrived damaged. It was a gift for my sister."

messages = [{"role": "user", "content": message_from_user_1}]

print("Добавление памяти...")
result = client.add(messages, user_id=user_id)
print(f"Результат: {result}")

# Задержка для индексации
time.sleep(2)

# Получение всех воспоминаний
print("\nВсе воспоминания пользователя:")
user_memories = client.get_all(user_id=user_id)
print(user_memories)

# Поиск релевантного контекста
print("\nПоиск по запросу 'replacement':")
new_message = "What's the status on the replacement?"
search_memory_response = client.search(query=new_message, user_id=user_id)
print(search_memory_response)

# Добавление с кастомной инструкцией (если поддерживается)
print("\nДобавление с кастомной инструкцией:")
try:
    advanced_memory_response = client.add(
        "I like to go hiking on weekends.",
        user_id=user_id,
        metadata={"category": "hobbies"},
        prompt="Extract hobbies specifically."
    )
    print(advanced_memory_response)
except Exception as e:
    print(f"Ошибка при добавлении с prompt: {e}")