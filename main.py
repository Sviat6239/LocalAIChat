import ollama
import json
import os
from math import ceil
import asyncio
from functools import lru_cache
import colorama
from concurrent.futures import ThreadPoolExecutor

colorama.init()

# Настройки
model_name = "qwen2.5-coder:3b"
memory_file = "chat_memory.json"
key_memory_file = "key_memories.json"
deep_memory_file = "deep_memory.json"
max_tokens = 65536
max_prompt_tokens = 16384
max_response_tokens = 16384
max_chat_tokens = 32768
max_deep_tokens = 16384
max_context_size = 8192
memory_window = 4

# Улучшенное системное сообщение
SYSTEM_PROMPT = {
    "role": "system",
    "content": "Ты дружелюбный и внимательный ИИ-помощник. Отвечай кратко, естественно и используй всю доступную информацию из текущей сессии (chat_history) и глубокой памяти (deep_memory) при ответах о пользователе. Если спрашивают 'как дела?', отвечай: 'У меня всё отлично, спасибо! А у тебя?'. Если спрашивают о планах, отвечай: 'Планирую помогать людям, как всегда! А ты что задумал?'. Если спрашивают о пользователе ('кто я?', 'что знаешь про меня?'), перечисляй всё, что известно из истории и глубокой памяти. Если данных нет, скажи: 'Я пока мало о тебе знаю, расскажи что-нибудь!'"
}

@lru_cache(maxsize=1000)
def estimate_tokens(text):
    return len(text) // 4 + 1

def split_text(text, max_tokens, type="prompt"):
    words = text.split()
    parts = []
    current_part = []
    current_tokens = 0

    for word in words:
        word_tokens = estimate_tokens(word)
        if current_tokens + word_tokens > max_tokens and current_part:
            parts.append(" ".join(current_part))
            current_part = [word]
            current_tokens = word_tokens
        else:
            current_part.append(word)
            current_tokens += word_tokens
    
    if current_part:
        parts.append(" ".join(current_part))
    
    return parts

def summarize_key_points(chat_history):
    summary = []
    for msg in chat_history:
        content = msg["content"].lower()
        if any(keyword in content for keyword in ["зовут", "люблю", "хочу", "дела", "планы"]):
            summary.append(msg)
    return summary[:memory_window]

async def split_context(chat_history, user_input_part, key_memories, deep_memory, max_tokens):
    full_context = f"{SYSTEM_PROMPT['content']}\n"
    if deep_memory:
        full_context += "Глубокая память:\n" + "\n".join(f"{msg['role']}: {msg['content']}" for msg in deep_memory) + "\n"
    if key_memories:
        full_context += "Ключевые моменты:\n" + "\n".join(f"{msg['role']}: {msg['content']}" for msg in key_memories) + "\n"
    full_context += "Текущий разговор:\n" + "\n".join(f"{msg['role']}: {msg['content']}" for msg in chat_history)
    full_context += f"\nuser: {user_input_part}"

    total_tokens = estimate_tokens(full_context)
    if total_tokens <= max_tokens:
        return [[SYSTEM_PROMPT] + deep_memory + key_memories + chat_history + [{"role": "user", "content": user_input_part}]], False

    parts = []
    current_part = [SYSTEM_PROMPT] + deep_memory.copy()
    current_tokens = estimate_tokens(SYSTEM_PROMPT["content"]) + sum(estimate_tokens(msg["content"]) for msg in deep_memory)
    split_occurred = False

    all_messages = key_memories + chat_history + [{"role": "user", "content": user_input_part}]
    for msg in all_messages:
        msg_tokens = estimate_tokens(msg["content"])
        if current_tokens + msg_tokens > max_tokens and len(current_part) > len(deep_memory) + 1:
            parts.append(current_part)
            current_part = [SYSTEM_PROMPT] + deep_memory.copy() + [msg]
            current_tokens = estimate_tokens(SYSTEM_PROMPT["content"]) + sum(estimate_tokens(m["content"]) for m in deep_memory) + msg_tokens
            split_occurred = True
        else:
            current_part.append(msg)
            current_tokens += msg_tokens
    
    if current_part:
        parts.append(current_part)

    return parts, split_occurred

async def load_memory(file_path, default=[]):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        try:
            if os.path.exists(file_path):
                result = await loop.run_in_executor(pool, lambda: json.load(open(file_path, "r", encoding="utf-8")))
                if isinstance(result, list):
                    return result
            return default
        except Exception as e:
            print(f"{colorama.Fore.RED}⚠️ Ошибка при загрузке {file_path}: {e}{colorama.Style.RESET_ALL}")
            return default

async def save_memory(memory, file_path, max_size=None):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        try:
            if max_size:
                total_tokens = sum(estimate_tokens(msg["content"]) for msg in memory)
                while total_tokens > max_size and memory:
                    memory.pop(0)
                    total_tokens = sum(estimate_tokens(msg["content"]) for msg in memory)
            await loop.run_in_executor(pool, lambda: json.dump(memory, open(file_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"{colorama.Fore.RED}⚠️ Ошибка при сохранении {file_path}: {e}{colorama.Style.RESET_ALL}")

async def process_prompt_part(prompt_part, chat_history, key_memories, deep_memory):
    prompt_tokens = estimate_tokens(prompt_part)
    print(f"{colorama.Fore.CYAN}ℹ️ Обработка части промпта, токенов: {prompt_tokens}{colorama.Style.RESET_ALL}")
    
    context_parts, was_split = await split_context(chat_history, prompt_part, key_memories, deep_memory, max_tokens)
    part_response = ""

    if not was_split:
        messages = context_parts[0]
        total_tokens = estimate_tokens("\n".join(f"{m['role']}: {m['content']}" for m in messages))
        print(f"{colorama.Fore.CYAN}ℹ️ Общий контекст: {total_tokens} токенов{colorama.Style.RESET_ALL}")
        try:
            response = await asyncio.to_thread(ollama.chat, model=model_name, messages=messages, options={"num_ctx": max_context_size})
            part_response = response["message"]["content"]
        except Exception as e:
            print(f"{colorama.Fore.RED}⚠️ Ошибка при обработке части промпта: {e}{colorama.Style.RESET_ALL}")
            part_response = "[Ошибка]"
    else:
        for i, part in enumerate(context_parts):
            total_tokens = estimate_tokens("\n".join(f"{m['role']}: {m['content']}" for m in part))
            print(f"{colorama.Fore.CYAN}�stainless Обработка части контекста {i + 1}/{len(context_parts)}, токенов: {total_tokens}{colorama.Style.RESET_ALL}")
            try:
                response = await asyncio.to_thread(ollama.chat, model=model_name, messages=part, options={"num_ctx": max_context_size})
                part_response += response["message"]["content"] + " "
            except Exception as e:
                print(f"{colorama.Fore.RED}⚠️ Ошибка при обработке части контекста {i + 1}: {e}{colorama.Style.RESET_ALL}")
                part_response += "[Ошибка] "

    return part_response

async def main():
    chat_history = await load_memory(memory_file)
    key_memories = await load_memory(key_memory_file)
    deep_memory = await load_memory(deep_memory_file)
    print(f"{colorama.Fore.GREEN}🤖 Запущен {model_name}. Введи 'выход', 'очистить', 'запомнить' или 'вспомнить'.{colorama.Style.RESET_ALL}")
    print(f"{colorama.Fore.CYAN}ℹ️ Max tokens: {max_tokens}, Max prompt tokens: {max_prompt_tokens}, Max response tokens: {max_response_tokens}, Max chat tokens: {max_chat_tokens}, Max deep tokens: {max_deep_tokens}, Context size: {max_context_size}{colorama.Style.RESET_ALL}")

    while True:
        try:
            user_input = input(f"{colorama.Fore.YELLOW}Ты: {colorama.Style.RESET_ALL}")

            if user_input.lower() in ["выход", "exit"]:
                print(f"{colorama.Fore.GREEN}🤖 Завершение работы...{colorama.Style.RESET_ALL}")
                await asyncio.gather(
                    save_memory(chat_history, memory_file, max_chat_tokens),
                    save_memory(key_memories, key_memory_file),
                    save_memory(deep_memory, deep_memory_file, max_deep_tokens)
                )
                break
            elif user_input.lower() in ["очистить", "clear"]:
                chat_history = []
                key_memories = []
                deep_memory = []
                await asyncio.gather(
                    save_memory(chat_history, memory_file),
                    save_memory(key_memories, key_memory_file),
                    save_memory(deep_memory, deep_memory_file)
                )
                print(f"{colorama.Fore.GREEN}🤖 Память очищена.{colorama.Style.RESET_ALL}")
                continue
            elif user_input.lower() == "запомнить":
                if chat_history:
                    summary = summarize_key_points(chat_history)
                    deep_memory.extend(summary)
                    await save_memory(deep_memory, deep_memory_file, max_deep_tokens)
                    print(f"{colorama.Fore.GREEN}🤖 Глубокая память обновлена:{colorama.Style.RESET_ALL} {[msg['content'] for msg in summary]}")
                else:
                    print(f"{colorama.Fore.YELLOW}🤖 Нет истории для запоминания.{colorama.Style.RESET_ALL}")
                continue
            elif user_input.lower() == "вспомнить":
                if chat_history or deep_memory:
                    print(f"{colorama.Fore.GREEN}🤖 Вот что я помню о тебе:{colorama.Style.RESET_ALL}")
                    if deep_memory:
                        print(f"{colorama.Fore.CYAN}Из глубокой памяти:{colorama.Style.RESET_ALL} {[msg['content'] for msg in deep_memory]}")
                    if chat_history:
                        user_info = [msg["content"] for msg in chat_history if msg["role"] == "user"]
                        print(f"{colorama.Fore.CYAN}Из текущей сессии:{colorama.Style.RESET_ALL} {user_info}")
                else:
                    print(f"{colorama.Fore.YELLOW}🤖 Пока ничего не знаю о тебе.{colorama.Style.RESET_ALL}")
                continue

            prompt_parts = split_text(user_input, max_prompt_tokens, type="prompt")
            combined_response = ""

            tasks = [process_prompt_part(part, chat_history, key_memories, deep_memory) for part in prompt_parts]
            responses = await asyncio.gather(*tasks)

            for part_response in responses:
                response_tokens = estimate_tokens(part_response)
                if response_tokens > max_response_tokens:
                    response_parts = split_text(part_response, max_response_tokens, type="response")
                    for k, resp_part in enumerate(response_parts):
                        resp_tokens = estimate_tokens(resp_part)
                        print(f"{colorama.Fore.GREEN}AI (часть {k + 1}/{len(response_parts)}, токенов: {resp_tokens}): {resp_part}{colorama.Style.RESET_ALL}")
                        combined_response += resp_part + " "
                else:
                    print(f"{colorama.Fore.GREEN}AI (токенов: {response_tokens}): {part_response}{colorama.Style.RESET_ALL}")
                    combined_response += part_response + " "

            combined_response = combined_response.strip()
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": combined_response})
            await save_memory(chat_history, memory_file, max_chat_tokens)

        except KeyboardInterrupt:
            print(f"\n{colorama.Fore.GREEN}🤖 Завершение работы по прерыванию...{colorama.Style.RESET_ALL}")
            await asyncio.gather(
                save_memory(chat_history, memory_file, max_chat_tokens),
                save_memory(key_memories, key_memory_file),
                save_memory(deep_memory, deep_memory_file, max_deep_tokens)
            )
            break
        except Exception as e:
            print(f"{colorama.Fore.RED}⚠️ Ошибка: {e}{colorama.Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())