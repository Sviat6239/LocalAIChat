import ollama
import json
import os
import asyncio
import aiohttp
import colorama
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import functools
import logging
from asyncio import Queue

colorama.init()

# Настройка асинхронного логгера
log_queue = Queue()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def log_writer():
    """Асинхронный обработчик логов из очереди"""
    while True:
        message = await log_queue.get()
        if message is None:  # Сигнал завершения
            break
        print(message, flush=True)
        log_queue.task_done()

# Запуск логгера в фоновом режиме
async def start_logger():
    asyncio.create_task(log_writer())

# Settings
model_name = "qwen2.5-coder:3b"
memory_file = "chat_memory.json"
key_memory_file = "key_memories.json"
deep_memory_file = "deep_memory.json"
compiled_memory_file = "compiled_memory.json"
system_prompt_file = "system_prompt.json"  # Новый файл для системного промпта
custom_settings_file = "custom_settings.json"
ollama_url = "http://localhost:11434/api/chat"
max_tokens = 65536
max_prompt_tokens = 16384
max_response_tokens = 16384
max_chat_tokens = 32768
max_deep_tokens = 16384
max_context_size = 8192
memory_window = 4
max_history_size = 50
compile_interval = 50
token_display_delay = 0.01  # Faster delay for smoother display

# Кэш для токенизации
@functools.lru_cache(maxsize=1000)
def cached_estimate_tokens(text):
    """Кэшированная версия estimate_tokens для ускорения повторных вычислений"""
    return len(text) // 4 + 1

# Загрузка системного промпта из файла
def load_system_prompt(file_path):
    """Loads system prompt from a JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        asyncio.create_task(log_queue.put(f"{colorama.Fore.RED}⚠️ System prompt file not found: {file_path}. Using default prompt.{colorama.Style.RESET_ALL}"))
        return {
            "role": "system",
            "content": "You are a concise, formal AI assistant. Respond logically using available data. How can I assist you?"
        }
    except json.JSONDecodeError:
        asyncio.create_task(log_queue.put(f"{colorama.Fore.RED}⚠️ Error decoding {file_path}. Using default prompt.{colorama.Style.RESET_ALL}"))
        return {
            "role": "system",
            "content": "You are a concise, formal AI assistant. Respond logically using available data. How can I assist you?"
        }

SYSTEM_PROMPT = load_system_prompt(system_prompt_file)

def load_custom_settings(file_path):
    """Loads custom user settings from a JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
            return settings.get("custom_instructions", "")
    except FileNotFoundError:
        asyncio.create_task(log_queue.put(f"{colorama.Fore.YELLOW}ℹ️ Custom settings file not found: {file_path}. Using default behavior.{colorama.Style.RESET_ALL}"))
        return ""
    except json.JSONDecodeError:
        asyncio.create_task(log_queue.put(f"{colorama.Fore.RED}⚠️ Error decoding {file_path}. Using default behavior.{colorama.Style.RESET_ALL}"))
        return ""

CUSTOM_INSTRUCTIONS = load_custom_settings(custom_settings_file)

async def type_text(text, color=colorama.Fore.GREEN, delay=0.02):
    """Prints text gradually with a delay between characters"""
    for char in text:
        print(color + char, end='', flush=True)
        await asyncio.sleep(delay)
    print(colorama.Style.RESET_ALL)

async def display_token(token, in_code_block=False, last_token_was_space=False):
    """Displays a token with proper spacing and code block handling"""
    if not token.strip() and not in_code_block:
        return last_token_was_space
    
    if token.startswith('```') or token.endswith('```'):
        if in_code_block and token.endswith('```'):
            print(colorama.Fore.GREEN + token, end='\n', flush=True)
            await asyncio.sleep(token_display_delay)
            return False
        elif not in_code_block and token.startswith('```'):
            print(colorama.Fore.GREEN + token, end='\n', flush=True)
            await asyncio.sleep(token_display_delay)
            return True
        return in_code_block

    if in_code_block:
        print(colorama.Fore.GREEN + token, end='', flush=True)
        await asyncio.sleep(token_display_delay)
        return token.endswith('\n') or token.endswith(' ')
    
    if last_token_was_space and not token.startswith(('.', ',', '!', '?', ':', ';', '\n', ' ')):
        print(colorama.Fore.GREEN + ' ', end='', flush=True)
    print(colorama.Fore.GREEN + token, end='', flush=True)
    await asyncio.sleep(token_display_delay)
    return token.endswith(' ') or token.endswith('\n')

def estimate_tokens(text):
    """Estimate token count with caching"""
    return cached_estimate_tokens(text)

def split_text(text, max_tokens, type="prompt"):
    """Splits text into parts based on max_tokens"""
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

def analyze_input_for_memory(text):
    """Analyzes user input to determine if it should go to key or deep memory"""
    text_lower = text.lower()
    deep_keywords = ["люблю", "хочу", "зовут", "мой", "мне", "предпочитаю", "обожаю", "тащусь", "love", "want", "called", "my", "i", "prefer"]
    key_keywords = ["сделал", "произошло", "важно", "событие", "did", "happened", "important", "event"]
    if text_lower.endswith("?") or any(q in text_lower for q in ["что", "как", "кто", "где", "когда", "почему", "what", "how", "who", "where", "when", "why"]):
        return None
    
    if any(keyword in text_lower for keyword in deep_keywords):
        return "deep"
    elif any(keyword in text_lower for keyword in key_keywords):
        return "key"
    if "я" in text_lower and any(id_word in text_lower for id_word in ["зовут", "called", "программист", "programmer"]):
        return "deep"
    return None

async def process_command_with_ai(user_input, chat_history):
    """Uses AI to interpret commands"""
    messages = [
        {"role": "system", "content": "You are a command interpreter. Identify if the input is a command (e.g., exit, clear, remember, recall, key, moment, messages, запомнить, вспомнить, очистить) and return it in the format: /command [args]. Return None if not a command."},
        {"role": "user", "content": f"Interpret this input: '{user_input}'"}
    ]
    try:
        response = await asyncio.to_thread(ollama.chat, model=model_name, messages=messages, options={"num_ctx": max_context_size})
        interpreted_command = response["message"]["content"].strip()
        return interpreted_command if interpreted_command.startswith("/") else None
    except Exception as e:
        await log_queue.put(f"{colorama.Fore.RED}⚠️ Error interpreting command: {e}{colorama.Style.RESET_ALL}")
        return None

# Оптимизация: фильтрация данных перед суммированием
def filter_redundant_info(info_list):
    """Фильтрует избыточные данные перед суммированием"""
    filtered = []
    seen_concepts = set()
    for item in info_list:
        key_parts = tuple(sorted(set(item.lower().split())))
        if key_parts not in seen_concepts:
            filtered.append(item)
            seen_concepts.add(key_parts)
    return filtered

async def summarize_user_info(deep_memory, key_memories, chat_history):
    """Summarizes user information from memory and chat history into a concise, formal description."""
    deep_info = [msg["content"] for msg in deep_memory if msg["role"] == "user"]
    key_info = [msg["content"] for msg in key_memories if msg["role"] == "user"]
    chat_info = [msg["content"] for msg in chat_history if msg["role"] == "user"]

    all_info = filter_redundant_info(deep_info + key_info + chat_info)
    if not all_info:
        return "I have limited information about you so far. Please share more about yourself."

    summary_prompt = (
        "Create a concise and formal summary about the user based on the following data: "
        f"{', '.join(all_info)}. "
        "Limit the summary to 1-2 sentences, exclude repetitive or minor details, focus on key facts, and present the information in a coherent manner."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant specialized in creating concise and accurate summaries. "
                "Respond formally, avoid slang, combine facts into a cohesive description, and keep it to 1-2 sentences."
            )
        },
        {"role": "user", "content": summary_prompt}
    ]

    try:
        response = await asyncio.to_thread(
            ollama.chat,
            model=model_name,
            messages=messages,
            options={"num_ctx": max_context_size}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        await log_queue.put(f"{colorama.Fore.RED}⚠️ Error summarizing user info: {e}{colorama.Style.RESET_ALL}")
        return "I have information about you, but I cannot summarize it right now."

# Оптимизация: сжатие контекста
def compress_context(context_parts):
    """Сжимает контекст, удаляя избыточные пробелы и повторы"""
    seen = set()
    compressed = []
    for part in context_parts:
        cleaned = " ".join(part.split())
        if cleaned not in seen:
            compressed.append(cleaned)
            seen.add(cleaned)
    return compressed

async def split_context(chat_history, user_input_part, key_memories, deep_memory, compiled_memory, max_tokens):
    context_parts = [SYSTEM_PROMPT["content"]]
    if CUSTOM_INSTRUCTIONS:
        context_parts.append(f"Custom Instructions: {CUSTOM_INSTRUCTIONS}")
    if deep_memory:
        context_parts.append("Deep Memory (preferences):")
        context_parts.extend(f"{msg['role']}: {msg['content']}" for msg in deep_memory)
    if key_memories:
        context_parts.append("Key Moments:")
        context_parts.extend(f"{msg['role']}: {msg['content']}" for msg in key_memories)
    if compiled_memory:
        context_parts.append("Compiled Memory:")
        context_parts.extend(f"{msg['role']}: {msg['content']}" for msg in compiled_memory)
    context_parts.append("Current Conversation:")
    context_parts.extend(f"{msg['role']}: {msg['content']}" for msg in chat_history[-max_history_size:])
    context_parts.append(f"user: {user_input_part}")

    # Оптимизация: сжатие контекста
    context_parts = compress_context(context_parts)

    total_tokens = estimate_tokens("\n".join(context_parts))
    if total_tokens <= max_tokens:
        return [[SYSTEM_PROMPT] + 
                ([{"role": "system", "content": f"Custom Instructions: {CUSTOM_INSTRUCTIONS}"}] if CUSTOM_INSTRUCTIONS else []) + 
                deep_memory + key_memories + compiled_memory + chat_history[-max_history_size:] + [{"role": "user", "content": user_input_part}]], False

    parts = []
    current_part = [SYSTEM_PROMPT] + ([{"role": "system", "content": f"Custom Instructions: {CUSTOM_INSTRUCTIONS}"}] if CUSTOM_INSTRUCTIONS else []) + deep_memory.copy()
    current_tokens = estimate_tokens(SYSTEM_PROMPT["content"]) + (estimate_tokens(CUSTOM_INSTRUCTIONS) if CUSTOM_INSTRUCTIONS else 0) + sum(estimate_tokens(msg["content"]) for msg in deep_memory)
    split_occurred = False

    all_messages = key_memories + compiled_memory + chat_history[-max_history_size:] + [{"role": "user", "content": user_input_part}]
    for msg in all_messages:
        msg_tokens = estimate_tokens(msg["content"])
        if current_tokens + msg_tokens > max_tokens and len(current_part) > len(deep_memory) + (1 if CUSTOM_INSTRUCTIONS else 0) + 1:
            parts.append(current_part)
            current_part = [SYSTEM_PROMPT] + ([{"role": "system", "content": f"Custom Instructions: {CUSTOM_INSTRUCTIONS}"}] if CUSTOM_INSTRUCTIONS else []) + deep_memory.copy() + [msg]
            current_tokens = estimate_tokens(SYSTEM_PROMPT["content"]) + (estimate_tokens(CUSTOM_INSTRUCTIONS) if CUSTOM_INSTRUCTIONS else 0) + sum(estimate_tokens(m["content"]) for msg in deep_memory) + msg_tokens
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
                    return result[:max_history_size] if file_path == memory_file else result
            return default
        except Exception as e:
            await log_queue.put(f"{colorama.Fore.RED}⚠️ Error loading {file_path}: {e}{colorama.Style.RESET_ALL}")
            return default

async def save_memory(memory, file_path, max_size=None):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        try:
            if max_size and file_path != compiled_memory_file:
                total_tokens = sum(estimate_tokens(msg["content"]) for msg in memory)
                while total_tokens > max_size and memory:
                    memory.pop(0)
                if file_path == memory_file:
                    memory = memory[-max_history_size:]
            await loop.run_in_executor(pool, lambda: json.dump(memory, open(file_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2))
        except Exception as e:
            await log_queue.put(f"{colorama.Fore.RED}⚠️ Error saving {file_path}: {e}{colorama.Style.RESET_ALL}")

# Оптимизация: пул потоков для обработки запросов к Ollama
ollama_pool = ThreadPoolExecutor(max_workers=4)

async def process_prompt_part(prompt_part, chat_history, key_memories, deep_memory, compiled_memory):
    prompt_tokens = estimate_tokens(prompt_part)
    await log_queue.put(f"ℹ️ Processing prompt part, tokens: {prompt_tokens}")

    context_parts, was_split = await split_context(chat_history, prompt_part, key_memories, deep_memory, compiled_memory, max_tokens)
    part_response = ""

    async with aiohttp.ClientSession() as session:
        if not was_split:
            messages = context_parts[0]
            total_tokens = estimate_tokens("\n".join(f"{m['role']}: {m['content']}" for m in messages))
            await log_queue.put(f"ℹ️ Total context: {total_tokens} tokens")

            if "запомни" in prompt_part.lower() or "remember" in prompt_part.lower():
                if len(chat_history) >= 2 and "свои слова" in prompt_part.lower():
                    last_response = chat_history[-2]["content"] if chat_history[-2]["role"] == "assistant" else None
                    if last_response:
                        key_memories.append({"role": "assistant", "content": last_response})
                        await save_memory(key_memories, key_memory_file)
                        await log_queue.put(f"🤖 Saved my last response to key memory: {last_response}")
                        return "OK"
                else:
                    key_memories.append({"role": "user", "content": prompt_part})
                    await save_memory(key_memories, key_memory_file)
                    await log_queue.put(f"🤖 Saved to key memory: {prompt_part}")
                    return "OK"

            if any(keyword in prompt_part.lower() for keyword in ["who am i", "what do you know", "what do you remember", "что ты знаешь", "кто я"]):
                summary = await summarize_user_info(deep_memory, key_memories, chat_history)
                response = f"Here is what I know about you: {summary}"
                await type_text(response, colorama.Fore.GREEN, delay=token_display_delay)
                return response.strip()

            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "options": {"num_ctx": max_context_size}
            }
            async with session.post(ollama_url, json=payload) as response:
                if response.status != 200:
                    await log_queue.put(f"⚠️ Error: HTTP {response.status}")
                    return "[Error]"
                
                response_text = ""
                in_code_block = False
                last_token_was_space = False
                async for line in response.content:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if "message" in chunk and "content" in chunk["message"]:
                            token = chunk["message"]["content"]
                            response_text += token
                            last_token_was_space = await display_token(token, in_code_block, last_token_was_space)
                            in_code_block = last_token_was_space if token.startswith('```') or token.endswith('```') else in_code_block
                    except json.JSONDecodeError:
                        continue
                part_response = response_text.strip()
        else:
            for i, part in enumerate(context_parts):
                total_tokens = estimate_tokens("\n".join(f"{m['role']}: {m['content']}" for m in part))
                await log_queue.put(f"ℹ️ Processing context part {i + 1}/{len(context_parts)}, tokens: {total_tokens}")
                
                payload = {
                    "model": model_name,
                    "messages": part,
                    "stream": True,
                    "options": {"num_ctx": max_context_size}
                }
                async with session.post(ollama_url, json=payload) as response:
                    if response.status != 200:
                        await log_queue.put(f"⚠️ Error: HTTP {response.status}")
                        part_response += "[Error] "
                        continue

                    response_text = ""
                    in_code_block = False
                    last_token_was_space = False
                    async for line in response.content:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if "message" in chunk and "content" in chunk["message"]:
                                token = chunk["message"]["content"]
                                response_text += token
                                last_token_was_space = await display_token(token, in_code_block, last_token_was_space)
                                in_code_block = last_token_was_space if token.startswith('```') or token.endswith('```') else in_code_block
                        except json.JSONDecodeError:
                            continue
                    part_response += response_text.strip() + " "

    print(colorama.Style.RESET_ALL)  # Reset color after streaming
    return part_response.strip()

async def main():
    await start_logger()  # Запуск асинхронного логгера

    chat_history = await load_memory(memory_file)
    key_memories = await load_memory(key_memory_file)
    deep_memory = await load_memory(deep_memory_file)
    compiled_memory = await load_memory(compiled_memory_file)
    message_count = len(chat_history) // 2

    await type_text(f"🤖 Started {model_name}. Commands: exit, clear, remember, remember_key_moment, recall, remember N messages (with or without /).", delay=0.02)
    await log_queue.put(f"ℹ️ Max tokens: {max_tokens}, Max prompt tokens: {max_prompt_tokens}, Max response tokens: {max_response_tokens}, Max chat tokens: {max_chat_tokens}, Max deep tokens: {max_deep_tokens}, Context size: {max_context_size}")
    if CUSTOM_INSTRUCTIONS:
        await log_queue.put(f"ℹ️ Custom instructions loaded: {CUSTOM_INSTRUCTIONS}")

    while True:
        try:
            user_input = input(f"{colorama.Fore.YELLOW}You: {colorama.Style.RESET_ALL}")
            input_lower = user_input.lower().strip()

            command = input_lower if input_lower.startswith("/") else await process_command_with_ai(user_input, chat_history)
            if command:
                if command.startswith("/exit") or command == "exit":
                    await type_text("🤖 Shutting down...", delay=0.02)
                    await asyncio.gather(
                        save_memory(chat_history, memory_file, max_chat_tokens),
                        save_memory(key_memories, key_memory_file),
                        save_memory(deep_memory, deep_memory_file, max_deep_tokens),
                        save_memory(compiled_memory, compiled_memory_file)
                    )
                    await log_queue.put(None)  # Сигнал завершения логгера
                    break
                elif command.startswith("/clear") or command == "clear":
                    chat_history = []
                    key_memories = []
                    deep_memory = []
                    compiled_memory = []
                    message_count = 0
                    await asyncio.gather(
                        save_memory(chat_history, memory_file),
                        save_memory(key_memories, key_memory_file),
                        save_memory(deep_memory, deep_memory_file),
                        save_memory(compiled_memory, compiled_memory_file)
                    )
                    await type_text("🤖 Memory cleared.", delay=0.02)
                elif command.startswith("/remember_key_moment") or command == "remember_key_moment":
                    if chat_history:
                        summary = summarize_key_points(chat_history, num_messages=3)
                        key_memories.extend(summary)
                        await save_memory(key_memories, key_memory_file)
                        await type_text(f"🤖 Key moments updated: {[msg['content'] for msg in summary]}", delay=0.02)
                    else:
                        await log_queue.put("🤖 No history to remember.")
                elif command.startswith("/remember") or command.startswith("remember"):
                    if "messages" in command:
                        try:
                            parts = command.split()
                            num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
                            if chat_history:
                                summary = summarize_key_points(chat_history, num_messages=num)
                                deep_memory.extend(summary)
                                await save_memory(deep_memory, deep_memory_file, max_deep_tokens)
                                await type_text(f"🤖 Deep memory updated ({num} messages): {[msg['content'] for msg in summary]}", delay=0.02)
                            else:
                                await log_queue.put("🤖 No history to remember.")
                        except ValueError:
                            await log_queue.put("⚠️ Specify a number, e.g., 'remember 5 messages'")
                    else:
                        if chat_history:
                            summary = summarize_key_points(chat_history)
                            deep_memory.extend(summary)
                            await save_memory(deep_memory, deep_memory_file, max_deep_tokens)
                            await type_text(f"🤖 Deep memory updated: {[msg['content'] for msg in summary]}", delay=0.02)
                        else:
                            await log_queue.put("🤖 No history to remember.")
                elif command.startswith("/recall") or command == "recall":
                    if chat_history or deep_memory or compiled_memory or key_memories:
                        await type_text("🤖 Here’s what I remember about you:", delay=0.02)
                        if deep_memory:
                            await type_text(f"From deep memory (preferences): {[msg['content'] for msg in deep_memory]}", colorama.Fore.CYAN, delay=0.02)
                        if key_memories:
                            await type_text(f"From key moments: {[msg['content'] for msg in key_memories]}", colorama.Fore.CYAN, delay=0.02)
                        if compiled_memory:
                            await type_text(f"From compiled memory: {[msg['content'] for msg in compiled_memory][-5:]}", colorama.Fore.CYAN, delay=0.02)
                        if chat_history:
                            user_info = [msg["content"] for msg in chat_history if msg["role"] == "user"]
                            await type_text(f"From current session: {user_info}", colorama.Fore.CYAN, delay=0.02)
                    else:
                        await log_queue.put("🤖 I don’t know anything about you yet.")
                continue

            memory_type = analyze_input_for_memory(user_input)
            if memory_type == "deep":
                deep_memory.append({"role": "user", "content": user_input})
                await save_memory(deep_memory, deep_memory_file, max_deep_tokens)
                await log_queue.put(f"🤖 Saved to deep memory: {user_input}")
            elif memory_type == "key":
                key_memories.append({"role": "user", "content": user_input})
                await save_memory(key_memories, key_memory_file)
                await log_queue.put(f"🤖 Saved to key memory: {user_input}")

            prompt_parts = split_text(user_input, max_prompt_tokens, type="prompt")
            combined_response = ""

            tasks = [process_prompt_part(part, chat_history, key_memories, deep_memory, compiled_memory) for part in prompt_parts]
            responses = await asyncio.gather(*tasks)

            for part_response in responses:
                combined_response += part_response + " "

            combined_response = combined_response.strip()
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": combined_response})
            message_count += 1

            if message_count >= compile_interval:
                new_compiled_block = compile_memory(chat_history, key_memories, compiled_memory)
                compiled_memory.extend(new_compiled_block)
                await save_memory(compiled_memory, compiled_memory_file)
                chat_history = chat_history[-max_history_size * 2:]
                message_count = 0
                await type_text(f"🤖 Memory compiled: {[msg['content'] for msg in new_compiled_block]}", delay=0.02)
            else:
                if len(chat_history) > max_history_size * 2:
                    chat_history = chat_history[-max_history_size * 2:]
                await save_memory(chat_history, memory_file, max_chat_tokens)

        except KeyboardInterrupt:
            await type_text("🤖 Shutting down due to interruption...", delay=0.02)
            await asyncio.gather(
                save_memory(chat_history, memory_file, max_chat_tokens),
                save_memory(key_memories, key_memory_file),
                save_memory(deep_memory, deep_memory_file, max_deep_tokens),
                save_memory(compiled_memory, compiled_memory_file)
            )
            await log_queue.put(None)  # Сигнал завершения логгера
            break
        except Exception as e:
            await log_queue.put(f"⚠️ Error: {e}")

def summarize_key_points(chat_history, num_messages=3):
    """Summarizes the last num_messages user messages"""
    user_messages = [msg for msg in chat_history if msg["role"] == "user"][-num_messages:]
    return user_messages[:min(num_messages, len(user_messages))]

def compile_memory(chat_history, key_memories, previous_compiled_memory):
    """Compiles chat_history and key_memories into compiled_memory with duplicate check"""
    compiled = []
    existing_contents = {msg["content"] for msg in previous_compiled_memory}
    keywords = Counter()

    for msg in chat_history:
        if msg["role"] == "user":
            compiled_content = f"User said: {msg['content']}"
            if compiled_content not in existing_contents:
                compiled.append({"role": "compiled", "content": compiled_content})
                words = msg["content"].lower().split()
                keywords.update([w for w in words if w in ["love", "want", "called"]])

    for msg in key_memories:
        if msg["content"] not in existing_contents:
            compiled.append({"role": "compiled", "content": msg["content"]})

    final_compiled = []
    seen_keywords = set()
    for entry in compiled:
        content_lower = entry["content"].lower()
        key_phrase = tuple(w for w in content_lower.split() if w in ["love", "want", "called"])
        if key_phrase not in seen_keywords or "User said" not in entry["content"]:
            final_compiled.append(entry)
            seen_keywords.add(key_phrase)

    return final_compiled

if __name__ == "__main__":
    asyncio.run(main())