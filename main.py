import ollama
import json
import os
import asyncio
import colorama
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

colorama.init()

# Settings
model_name = "qwen2.5-coder:3b"
memory_file = "chat_memory.json"
key_memory_file = "key_memories.json"
deep_memory_file = "deep_memory.json"
compiled_memory_file = "compiled_memory.json"
system_prompt_file = "system_prompt.json"
max_tokens = 65536
max_prompt_tokens = 16384
max_response_tokens = 16384
max_chat_tokens = 32768
max_deep_tokens = 16384
max_context_size = 8192
memory_window = 4
max_history_size = 50
compile_interval = 50

def load_system_prompt(file_path):
    """Loads system prompt from a JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{colorama.Fore.RED}⚠️ System prompt file not found: {file_path}{colorama.Style.RESET_ALL}")
        return {"role": "system", "content": "Default assistant prompt"}

SYSTEM_PROMPT = load_system_prompt(system_prompt_file)

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

def summarize_key_points(chat_history, num_messages=3):
    """Summarizes the last num_messages user messages"""
    user_messages = [msg for msg in chat_history if msg["role"] == "user"][-num_messages:]
    return user_messages[:min(num_messages, len(user_messages))]

def compile_memory(chat_history, key_memories, previous_compiled_memory):
    """Compiles chat_history and key_memories into compiled_memory with duplicate check"""
    compiled = []
    existing_contents = {msg["content"] for msg in previous_compiled_memory}
    keywords = Counter()

    # Collect from chat_history
    for msg in chat_history:
        if msg["role"] == "user":
            compiled_content = f"User said: {msg['content']}"
            if compiled_content not in existing_contents:
                compiled.append({"role": "compiled", "content": compiled_content})
                words = msg["content"].lower().split()
                keywords.update([w for w in words if w in ["love", "want", "called"]])

    # Add key_memories, excluding duplicates
    for msg in key_memories:
        if msg["content"] not in existing_contents:
            compiled.append({"role": "compiled", "content": msg["content"]})

    # Exclude duplicates based on key phrases
    final_compiled = []
    seen_keywords = set()
    for entry in compiled:
        content_lower = entry["content"].lower()
        key_phrase = tuple(w for w in content_lower.split() if w in ["love", "want", "called"])
        if key_phrase not in seen_keywords or "User said" not in entry["content"]:
            final_compiled.append(entry)
            seen_keywords.add(key_phrase)

    return final_compiled

async def split_context(chat_history, user_input_part, key_memories, deep_memory, compiled_memory, max_tokens):
    context_parts = [SYSTEM_PROMPT["content"]]
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

    total_tokens = estimate_tokens("\n".join(context_parts))
    if total_tokens <= max_tokens:
        return [[SYSTEM_PROMPT] + deep_memory + key_memories + compiled_memory + chat_history[-max_history_size:] + [{"role": "user", "content": user_input_part}]], False

    parts = []
    current_part = [SYSTEM_PROMPT] + deep_memory.copy()
    current_tokens = estimate_tokens(SYSTEM_PROMPT["content"]) + sum(estimate_tokens(msg["content"]) for msg in deep_memory)
    split_occurred = False

    all_messages = key_memories + compiled_memory + chat_history[-max_history_size:] + [{"role": "user", "content": user_input_part}]
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
                    return result[:max_history_size] if file_path == memory_file else result
            return default
        except Exception as e:
            print(f"{colorama.Fore.RED}⚠️ Error loading {file_path}: {e}{colorama.Style.RESET_ALL}")
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
            print(f"{colorama.Fore.RED}⚠️ Error saving {file_path}: {e}{colorama.Style.RESET_ALL}")

async def process_prompt_part(prompt_part, chat_history, key_memories, deep_memory, compiled_memory):
    prompt_tokens = estimate_tokens(prompt_part)
    print(f"{colorama.Fore.CYAN}ℹ️ Processing prompt part, tokens: {prompt_tokens}{colorama.Style.RESET_ALL}")
    
    context_parts, was_split = await split_context(chat_history, prompt_part, key_memories, deep_memory, compiled_memory, max_tokens)
    part_response = ""

    if not was_split:
        messages = context_parts[0]
        total_tokens = estimate_tokens("\n".join(f"{m['role']}: {m['content']}" for m in messages))
        print(f"{colorama.Fore.CYAN}ℹ️ Total context: {total_tokens} tokens{colorama.Style.RESET_ALL}")
        
        if any(keyword in prompt_part.lower() for keyword in ["who am i", "what do you know", "what do you remember"]):
            user_info = "\nKnown about user:\n"
            for msg in deep_memory + chat_history:
                if msg["role"] == "user" and any(kw in msg["content"].lower() for kw in ["called", "love", "want"]):
                    user_info += f"- {msg['content']}\n"
            messages.insert(-1, {"role": "system", "content": user_info if user_info.strip() != "Known about user:" else "I don’t know much about you yet, tell me something!"})

        try:
            response = await asyncio.to_thread(ollama.chat, model=model_name, messages=messages, options={"num_ctx": max_context_size})
            part_response = response["message"]["content"]
        except Exception as e:
            print(f"{colorama.Fore.RED}⚠️ Error processing prompt part: {e}{colorama.Style.RESET_ALL}")
            part_response = "[Error]"
    else:
        for i, part in enumerate(context_parts):
            total_tokens = estimate_tokens("\n".join(f"{m['role']}: {m['content']}" for m in part))
            print(f"{colorama.Fore.CYAN}ℹ️ Processing context part {i + 1}/{len(context_parts)}, tokens: {total_tokens}{colorama.Style.RESET_ALL}")
            try:
                response = await asyncio.to_thread(ollama.chat, model=model_name, messages=part, options={"num_ctx": max_context_size})
                part_response += response["message"]["content"] + " "
            except Exception as e:
                print(f"{colorama.Fore.RED}⚠️ Error processing context part {i + 1}: {e}{colorama.Style.RESET_ALL}")
                part_response += "[Error] "

    return part_response

async def main():
    chat_history = await load_memory(memory_file)
    key_memories = await load_memory(key_memory_file)
    deep_memory = await load_memory(deep_memory_file)
    compiled_memory = await load_memory(compiled_memory_file)
    message_count = len(chat_history) // 2

    print(f"{colorama.Fore.GREEN}🤖 Started {model_name}. Enter 'exit', 'clear', 'remember', 'remember key moment', or 'recall'.{colorama.Style.RESET_ALL}")
    print(f"{colorama.Fore.CYAN}ℹ️ Max tokens: {max_tokens}, Max prompt tokens: {max_prompt_tokens}, Max response tokens: {max_response_tokens}, Max chat tokens: {max_chat_tokens}, Max deep tokens: {max_deep_tokens}, Context size: {max_context_size}{colorama.Style.RESET_ALL}")

    while True:
        try:
            user_input = input(f"{colorama.Fore.YELLOW}You: {colorama.Style.RESET_ALL}")

            if user_input.lower() in ["exit"]:
                print(f"{colorama.Fore.GREEN}🤖 Shutting down...{colorama.Style.RESET_ALL}")
                await asyncio.gather(
                    save_memory(chat_history, memory_file, max_chat_tokens),
                    save_memory(key_memories, key_memory_file),
                    save_memory(deep_memory, deep_memory_file, max_deep_tokens),
                    save_memory(compiled_memory, compiled_memory_file)
                )
                break
            elif user_input.lower() in ["clear"]:
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
                print(f"{colorama.Fore.GREEN}🤖 Memory cleared.{colorama.Style.RESET_ALL}")
                continue
            elif user_input.lower() == "remember":
                if chat_history:
                    summary = summarize_key_points(chat_history)
                    deep_memory.extend(summary)  # To deep memory
                    await save_memory(deep_memory, deep_memory_file, max_deep_tokens)
                    print(f"{colorama.Fore.GREEN}🤖 Deep memory updated:{colorama.Style.RESET_ALL} {[msg['content'] for msg in summary]}")
                else:
                    print(f"{colorama.Fore.YELLOW}🤖 No history to remember.{colorama.Style.RESET_ALL}")
                continue
            elif user_input.lower() == "remember key moment":
                if chat_history:
                    summary = summarize_key_points(chat_history, num_messages=3)
                    key_memories.extend(summary)
                    await save_memory(key_memories, key_memory_file)
                    print(f"{colorama.Fore.GREEN}🤖 Key moments updated:{colorama.Style.RESET_ALL} {[msg['content'] for msg in summary]}")
                else:
                    print(f"{colorama.Fore.YELLOW}🤖 No history to remember.{colorama.Style.RESET_ALL}")
                continue
            elif user_input.lower().startswith("remember ") and "messages" in user_input.lower():
                try:
                    num = int(user_input.split()[1])
                    if chat_history:
                        summary = summarize_key_points(chat_history, num_messages=num)
                        key_memories.extend(summary)
                        await save_memory(key_memories, key_memory_file)
                        print(f"{colorama.Fore.GREEN}🤖 Key moments updated ({num} messages):{colorama.Style.RESET_ALL} {[msg['content'] for msg in summary]}")
                    else:
                        print(f"{colorama.Fore.YELLOW}🤖 No history to remember.{colorama.Style.RESET_ALL}")
                except ValueError:
                    print(f"{colorama.Fore.RED}⚠️ Specify a number, e.g., 'remember 5 messages'{colorama.Style.RESET_ALL}")
                continue
            elif user_input.lower() == "recall":
                if chat_history or deep_memory or compiled_memory or key_memories:
                    print(f"{colorama.Fore.GREEN}🤖 Here’s what I remember about you:{colorama.Style.RESET_ALL}")
                    if deep_memory:
                        print(f"{colorama.Fore.CYAN}From deep memory (preferences):{colorama.Style.RESET_ALL} {[msg['content'] for msg in deep_memory]}")
                    if key_memories:
                        print(f"{colorama.Fore.CYAN}From key moments:{colorama.Style.RESET_ALL} {[msg['content'] for msg in key_memories]}")
                    if compiled_memory:
                        print(f"{colorama.Fore.CYAN}From compiled memory:{colorama.Style.RESET_ALL} {[msg['content'] for msg in compiled_memory][-5:]}")
                    if chat_history:
                        user_info = [msg["content"] for msg in chat_history if msg["role"] == "user"]
                        print(f"{colorama.Fore.CYAN}From current session:{colorama.Style.RESET_ALL} {user_info}")
                else:
                    print(f"{colorama.Fore.YELLOW}🤖 I don’t know anything about you yet.{colorama.Style.RESET_ALL}")
                continue

            prompt_parts = split_text(user_input, max_prompt_tokens, type="prompt")
            combined_response = ""

            tasks = [process_prompt_part(part, chat_history, key_memories, deep_memory, compiled_memory) for part in prompt_parts]
            responses = await asyncio.gather(*tasks)

            for part_response in responses:
                response_tokens = estimate_tokens(part_response)
                if response_tokens > max_response_tokens:
                    response_parts = split_text(part_response, max_response_tokens, type="response")
                    for k, resp_part in enumerate(response_parts):
                        resp_tokens = estimate_tokens(resp_part)
                        print(f"{colorama.Fore.GREEN}AI (part {k + 1}/{len(response_parts)}, tokens: {resp_tokens}): {resp_part}{colorama.Style.RESET_ALL}")
                        combined_response += resp_part + " "
                else:
                    print(f"{colorama.Fore.GREEN}AI (tokens: {response_tokens}): {part_response}{colorama.Style.RESET_ALL}")
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
                print(f"{colorama.Fore.GREEN}🤖 Memory compiled:{colorama.Style.RESET_ALL} {[msg['content'] for msg in new_compiled_block]}")
            else:
                if len(chat_history) > max_history_size * 2:
                    chat_history = chat_history[-max_history_size * 2:]
                await save_memory(chat_history, memory_file, max_chat_tokens)

        except KeyboardInterrupt:
            print(f"\n{colorama.Fore.GREEN}🤖 Shutting down due to interruption...{colorama.Style.RESET_ALL}")
            await asyncio.gather(
                save_memory(chat_history, memory_file, max_chat_tokens),
                save_memory(key_memories, key_memory_file),
                save_memory(deep_memory, deep_memory_file, max_deep_tokens),
                save_memory(compiled_memory, compiled_memory_file)
            )
            break
        except Exception as e:
            print(f"{colorama.Fore.RED}⚠️ Error: {e}{colorama.Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())