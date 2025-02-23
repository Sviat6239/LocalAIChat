package utils

import (
	"strings"
	"sync"

	"yourproject/config" // Замените на реальный путь
)

// Функция для оценки токенов (замена tiktoken, так как в Go нет прямого эквивалента; можно использовать стороннюю библиотеку)
func estimateTokens(text string) int {
	// Простая замена: считаем слова как токены (реальная реализация может потребовать интеграции с tiktoken или другой библиотекой)
	words := strings.Fields(text)
	return len(words)
}

// Кэширование с использованием sync.Map, так как Go не имеет встроенного lru_cache как Python
var tokenCache sync.Map

func cachedEstimateTokens(text string) int {
	if cached, ok := tokenCache.Load(text); ok {
		return cached.(int)
	}
	tokens := estimateTokens(text)
	tokenCache.Store(text, tokens)
	return tokens
}

func SplitText(text string, maxTokens int, _type string) []string {
	words := strings.Split(text, " ")
	parts := make([]string, 0)
	currentPart := make([]string, 0)
	currentTokens := 0

	for _, word := range words {
		wordTokens := cachedEstimateTokens(word)
		if currentTokens+wordTokens > maxTokens && len(currentPart) > 0 {
			parts = append(parts, strings.Join(currentPart, " "))
			currentPart = []string{word}
			currentTokens = wordTokens
		} else {
			currentPart = append(currentPart, word)
			currentTokens += wordTokens
		}
	}

	if len(currentPart) > 0 {
		parts = append(parts, strings.Join(currentPart, " "))
	}

	return parts
}

func CompressContext(contextParts []string) []string {
	seen := make(map[string]bool)
	compressed := make([]string, 0)
	for _, part := range contextParts {
		cleaned := strings.Join(strings.Fields(part), " ")
		if !seen[cleaned] {
			compressed = append(compressed, cleaned)
			seen[cleaned] = true
		}
	}
	return compressed
}