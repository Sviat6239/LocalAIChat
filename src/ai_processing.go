package ai_processing

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"yourproject/config"    // Замените на реальный путь
	"yourproject/display"  // Замените на реальный путь
	"yourproject/logging_setup"
	"yourproject/memory"
	"yourproject/utils"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func ProcessCommandWithAI(userInput string, chatHistory []Message) string {
	// Подготовка сообщений для интерпретации команды
	messages := []Message{
		{Role: "system", Content: "You are a command interpreter. Identify if the input is a command (e.g., exit, clear, remember, recall, key, moment, messages, запомнить, вспомнить, очистить) and return it in the format: /command [args]. Return None if not a command."},
		{Role: "user", Content: fmt.Sprintf("Interpret this input: '%s'", userInput)},
	}

	// Вызов Ollama через HTTP (асинхронность эмулируется через горутины)
	resp, err := callOllama(messages, config.MODEL_NAME, config.MAX_CONTEXT_SIZE)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error interpreting command: %v\033[0m", err)
		return ""
	}

	interpretedCommand := strings.TrimSpace(resp)
	if strings.HasPrefix(interpretedCommand, "/") {
		return interpretedCommand
	}
	return ""
}

func SummarizeUserInfo(deepMemory, keyMemories, chatHistory []Message) string {
	// Извлечение информации из памяти
	deepInfo := make([]string, 0)
	for _, msg := range deepMemory {
		if msg.Role == "user" {
			deepInfo = append(deepInfo, msg.Content)
		}
	}
	keyInfo := make([]string, 0)
	for _, msg := range keyMemories {
		if msg.Role == "user" {
			keyInfo = append(keyInfo, msg.Content)
		}
	}
	chatInfo := make([]string, 0)
	for _, msg := range chatHistory {
		if msg.Role == "user" {
			chatInfo = append(chatInfo, msg.Content)
		}
	}

	allInfo := memory.FilterRedundantInfo(append(append(deepInfo, keyInfo...), chatInfo...))
	if len(allInfo) == 0 {
		return "I have limited information about you so far. Please share more about yourself."
	}

	summaryPrompt := fmt.Sprintf(
		"Create a concise and formal summary about the user based on the following data: %s. Limit the summary to 1-2 sentences, exclude repetitive or minor details, focus on key facts, and present the information in a coherent manner.",
		strings.Join(allInfo, ", "),
	)
	messages := []Message{
		{
			Role: "system",
			Content: "You are an assistant specialized in creating concise and accurate summaries. Respond formally, avoid slang, combine facts into a cohesive description, and keep it to 1-2 sentences.",
		},
		{Role: "user", Content: summaryPrompt},
	}

	resp, err := callOllama(messages, config.MODEL_NAME, config.MAX_CONTEXT_SIZE)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprint("\033[31m⚠️ Error summarizing user info: %v\033[0m", err)
		return "I have information about you, but I cannot summarize it right now."
	}
	return strings.TrimSpace(resp)
}

func ProcessPromptPart(promptPart string, chatHistory, keyMemories, deepMemory, compiledMemory []Message) (string, error) {
	promptTokens := utils.EstimateTokens(promptPart)
	logging_setup.LogQueue <- fmt.Sprintf("ℹ️ Processing prompt part, tokens: %d", promptTokens)

	// Разделение контекста
	contextParts, wasSplit, err := memory.SplitContext(chatHistory, promptPart, keyMemories, deepMemory, compiledMemory, config.MAX_TOKENS)
	if err != nil {
		return "", err
	}
	partResponse := ""

	if !wasSplit {
		messages := contextParts[0]
		totalTokens := utils.EstimateTokens(strings.Join(flattenMessages(messages), "\n"))
		logging_setup.LogQueue <- fmt.Sprintf("ℹ️ Total context: %d tokens", totalTokens)

		// Проверка на команды "remember" или "запомни"
		promptLower := strings.ToLower(promptPart)
		if strings.Contains(promptLower, "запомни") || strings.Contains(promptLower, "remember") {
			if len(chatHistory) >= 2 && strings.Contains(promptLower, "свои слова") {
				lastResponse := ""
				if chatHistory[len(chatHistory)-2].Role == "assistant" {
					lastResponse = chatHistory[len(chatHistory)-2].Content
				}
				if lastResponse != "" {
					keyMemories = append(keyMemories, Message{Role: "assistant", Content: lastResponse})
					memory.SaveMemory(keyMemories, "key_memories", 0)
					logging_setup.LogQueue <- fmt.Sprintf("🤖 Saved my last response to key memory: %s", lastResponse)
					return "OK", nil
				}
			} else {
				keyMemories = append(keyMemories, Message{Role: "user", Content: promptPart})
				memory.SaveMemory(keyMemories, "key_memories", 0)
				logging_setup.LogQueue <- fmt.Sprintf("🤖 Saved to key memory: %s", promptPart)
				return "OK", nil
			}
		}

		// Проверка на запросы о памяти
		keywords := []string{"who am i", "what do you know", "what do you remember", "что ты знаешь", "кто я"}
		for _, keyword := range keywords {
			if strings.Contains(promptLower, keyword) {
				summary := SummarizeUserInfo(deepMemory, keyMemories, chatHistory)
				response := fmt.Sprintf("Here is what I know about you: %s", summary)
				display.TypeText(response, "\033[32m", config.TOKEN_DISPLAY_DELAY)
				return response, nil
			}
		}

		// Вызов Ollama с потоковым ответом
		payload := map[string]interface{}{
			"model":    config.MODEL_NAME,
			"messages": messages,
			"stream":   true,
			"options": map[string]int{
				"num_ctx": config.MAX_CONTEXT_SIZE,
			},
		}
		resp, err := streamOllama(payload)
		if err != nil {
			logging_setup.LogQueue <- "⚠️ Error: HTTP error"
			return "[Error]", err
		}
		return strings.TrimSpace(resp), nil
	} else {
		var wg sync.WaitGroup
		responses := make(chan string, len(contextParts))
		for i, part := range contextParts {
			wg.Add(1)
			go func(i int, part []Message) {
				defer wg.Done()
				totalTokens := utils.EstimateTokens(strings.Join(flattenMessages(part), "\n"))
				logging_setup.LogQueue <- fmt.Sprintf("ℹ️ Processing context part %d/%d, tokens: %d", i+1, len(contextParts), totalTokens)

				payload := map[string]interface{}{
					"model":    config.MODEL_NAME,
					"messages": part,
					"stream":   true,
					"options": map[string]int{
						"num_ctx": config.MAX_CONTEXT_SIZE,
					},
				}
				resp, err := streamOllama(payload)
				if err != nil {
					logging_setup.LogQueue <- fmt.Sprintf("⚠️ Error: HTTP error for part %d", i+1)
					responses <- "[Error] "
					return
				}
				responses <- strings.TrimSpace(resp) + " "
			}(i, part)
		}

		go func() {
			wg.Wait()
			close(responses)
		}()

		for resp := range responses {
			partResponse += resp
		}
		return strings.TrimSpace(partResponse), nil
	}
}

// Вспомогательная функция для вызова Ollama через HTTP
func callOllama(messages []Message, model string, contextSize int) (string, error) {
	payload := map[string]interface{}{
		"model":    model,
		"messages": messages,
		"stream":   false,
		"options": map[string]int{
			"num_ctx": contextSize,
		},
	}
	jsonData, _ := json.Marshal(payload)

	resp, err := http.Post(config.OLLAMA_URL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	if message, ok := result["message"].(map[string]interface{}); ok {
		if content, ok := message["content"].(string); ok {
			return content, nil
		}
	}
	return "", fmt.Errorf("invalid response format")
}

// Вспомогательная функция для потокового вызова Ollama (эмуляция стриминга)
func streamOllama(payload map[string]interface{}) (string, error) {
	jsonData, _ := json.Marshal(payload)
	resp, err := http.Post(config.OLLAMA_URL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	responseText := ""
	inCodeBlock := false
	lastTokenWasSpace := false

	decoder := json.NewDecoder(resp.Body)
	for {
		var chunk map[string]interface{}
		if err := decoder.Decode(&chunk); err == io.EOF {
			break
		} else if err != nil {
			continue
		}

		if message, ok := chunk["message"].(map[string]interface{}); ok {
			if content, ok := message["content"].(string); ok {
				token := content
				responseText += token
				lastTokenWasSpace = displayToken(token, inCodeBlock, lastTokenWasSpace)
				inCodeBlock = lastTokenWasSpace // Простая эмуляция блока кода
				time.Sleep(time.Duration(config.TOKEN_DISPLAY_DELAY * 1000 * 1000 * 1000)) // Задержка в наносекундах
			}
		}
	}
	return responseText, nil
}

// Вспомогательная функция для отображения токена (эмуляция display_token)
func displayToken(token string, inCodeBlock bool, lastTokenWasSpace bool) bool {
	// Простая эмуляция: выводим токен в терминал с небольшой задержкой
	fmt.Print(token)
	if strings.HasPrefix(token, "```") || strings.HasSuffix(token, "```") {
		return !lastTokenWasSpace
	}
	return strings.TrimSpace(token) == ""
}

// Вспомогательная функция для объединения сообщений в строку
func flattenMessages(messages []Message) []string {
	result := make([]string, 0, len(messages))
	for _, msg := range messages {
		result = append(result, fmt.Sprintf("%s: %s", msg.Role, msg.Content))
	}
	return result
}