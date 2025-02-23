package config

import (
	"encoding/json"
)


const (
	MODEL_NAME         = "qwen2.5-coder:3b"
	MEMORY_DB          = "memory.db"
	SYSTEM_PROMPT_FILE = "system_prompt.json"
	CUSTOM_SETTINGS_FILE = "custom_settings.json"
	OLLAMA_URL         = "http://localhost:11434/api/chat"
)

// Ограничения токенов и других параметров
const (
	MAX_TOKENS         = 65536
	MAX_PROMPT_TOKENS  = 16384
	MAX_RESPONSE_TOKENS = 16384
	MAX_CHAT_TOKENS    = 32768
	MAX_DEEP_TOKENS    = 16384
	MAX_CONTEXT_SIZE   = 8192
	MEMORY_WINDOW      = 4
	MAX_HISTORY_SIZE   = 50
	COMPILE_INTERVAL   = 50
	TOKEN_DISPLAY_DELAY = 0.01
)

// Структура для хранения системного промпта по умолчанию
var DEFAULT_SYSTEM_PROMPT = struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}{
	Role:    "system",
	Content: "You are a concise, formal AI assistant. Respond logically using available data. How can I assist you?",
}