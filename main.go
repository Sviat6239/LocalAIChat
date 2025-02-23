package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"src/config" 
	"src/display" 
	"src/ai_processing"
	"src/logging_setup"
	"src/memory"
	"src/utils"
)

var (
	DEFAULT_SYSTEM_PROMPT string
	CUSTOM_INSTRUCTIONS   string
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func loadSystemPrompt(filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ System prompt file not found: %s. Using default prompt.\033[0m", filePath)
		return config.DEFAULT_SYSTEM_PROMPT, nil // Используем значение по умолчанию
	}

	var prompt string
	if err := json.Unmarshal(data, &prompt); err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error decoding %s. Using default prompt.\033[0m", filePath)
		return config.DEFAULT_SYSTEM_PROMPT, nil
	}
	return prompt, nil
}

func loadCustomSettings(filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[33mℹ️ Custom settings file not found: %s. Using default behavior.\033[0m", filePath)
		return "", nil
	}

	var settings map[string]interface{}
	if err := json.Unmarshal(data, &settings); err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error decoding %s. Using default behavior.\033[0m", filePath)
		return "", nil
	}

	if instructions, ok := settings["custom_instructions"].(string); ok {
		return instructions, nil
	}
	return "", nil
}

func main() {
	var wg sync.WaitGroup
	logging_setup.StartLogger()

	// Загрузка системного промпта и пользовательских настроек
	var systemPromptErr, customSettingsErr error
	wg.Add(2)
	go func() {
		defer wg.Done()
		prompt, err := loadSystemPrompt(config.SYSTEM_PROMPT_FILE)
		if err != nil {
			fmt.Printf("Error loading system prompt: %v\n", err)
			return
		}
		DEFAULT_SYSTEM_PROMPT = prompt
	}()

	go func() {
		defer wg.Done()
		instructions, err := loadCustomSettings(config.CUSTOM_SETTINGS_FILE)
		if err != nil {
			fmt.Printf("Error loading custom settings: %v\n", err)
			return
		}
		CUSTOM_INSTRUCTIONS = instructions
	}()
	wg.Wait()

	// Загрузка памяти
	chatHistory, _ := memory.LoadMemory("chat_history")
	keyMemories, _ := memory.LoadMemory("key_memories")
	deepMemory, _ := memory.LoadMemory("deep_memory")
	compiledMemory, _ := memory.LoadMemory("compiled_memory")
	messageCount := len(chatHistory) / 2

	display.TypeText(fmt.Sprintf("🤖 Started %s. Commands: exit, clear, remember, remember_key_moment, recall, remember N messages (with or without /).", config.MODEL_NAME), 0.02)
	logging_setup.LogQueue <- fmt.Sprintf("ℹ️ Max tokens: %d, Max prompt tokens: %d, Max response tokens: %d, Max chat tokens: %d, Max deep tokens: %d, Context size: %d",
		config.MAX_TOKENS, config.MAX_PROMPT_TOKENS, config.MAX_RESPONSE_TOKENS, config.MAX_CHAT_TOKENS, config.MAX_DEEP_TOKENS, config.MAX_CONTEXT_SIZE)
	if CUSTOM_INSTRUCTIONS != "" {
		logging_setup.LogQueue <- fmt.Sprintf("ℹ️ Custom instructions loaded: %s", CUSTOM_INSTRUCTIONS)
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\033[33mYou: \033[0m")
		if !scanner.Scan() {
			break
		}
		userInput := scanner.Text()
		inputLower := strings.ToLower(strings.TrimSpace(userInput))

		var command string
		if strings.HasPrefix(inputLower, "/") {
			command = inputLower
		} else {
			command = ai_processing.ProcessCommandWithAI(userInput, chatHistory)
		}

		if command != "" {
			switch {
			case strings.HasPrefix(command, "/exit") || command == "exit":
				display.TypeText("🤖 Shutting down...", 0.02)
				var saveWg sync.WaitGroup
				saveWg.Add(4)
				go func() { defer saveWg.Done(); memory.SaveMemory(chatHistory, "chat_history", config.MAX_CHAT_TOKENS) }()
				go func() { defer saveWg.Done(); memory.SaveMemory(keyMemories, "key_memories", 0) }()
				go func() { defer saveWg.Done(); memory.SaveMemory(deepMemory, "deep_memory", config.MAX_DEEP_TOKENS) }()
				go func() { defer saveWg.Done(); memory.SaveMemory(compiledMemory, "compiled_memory", 0) }()
				saveWg.Wait()
				logging_setup.LogQueue <- nil
				return
			case strings.HasPrefix(command, "/clear") || command == "clear":
				chatHistory = []Message{}
				keyMemories = []Message{}
				deepMemory = []Message{}
				compiledMemory = []Message{}
				messageCount = 0
				var clearWg sync.WaitGroup
				clearWg.Add(4)
				go func() { defer clearWg.Done(); memory.SaveMemory(chatHistory, "chat_history", 0) }()
				go func() { defer clearWg.Done(); memory.SaveMemory(keyMemories, "key_memories", 0) }()
				go func() { defer clearWg.Done(); memory.SaveMemory(deepMemory, "deep_memory", 0) }()
				go func() { defer clearWg.Done(); memory.SaveMemory(compiledMemory, "compiled_memory", 0) }()
				clearWg.Wait()
				display.TypeText("🤖 Memory cleared.", 0.02)
			case strings.HasPrefix(command, "/remember_key_moment") || command == "remember_key_moment":
				if len(chatHistory) > 0 {
					summary := memory.SummarizeKeyPoints(chatHistory, 3)
					keyMemories = append(keyMemories, summary...)
					memory.SaveMemory(keyMemories, "key_memories", 0)
					contents := make([]string, 0, len(summary))
					for _, msg := range summary {
						contents = append(contents, msg.Content)
					}
					display.TypeText(fmt.Sprintf("🤖 Key moments updated: %v", contents), 0.02)
				} else {
					logging_setup.LogQueue <- "🤖 No history to remember."
				}
			case strings.HasPrefix(command, "/remember") || strings.HasPrefix(command, "remember"):
				if strings.Contains(command, "messages") {
					parts := strings.Fields(command)
					var num int
					if len(parts) > 1 && isDigit(parts[1]) {
						num = parseInt(parts[1])
					} else {
						num = 3
					}
					if len(chatHistory) > 0 {
						summary := memory.SummarizeKeyPoints(chatHistory, num)
						deepMemory = append(deepMemory, summary...)
						memory.SaveMemory(deepMemory, "deep_memory", config.MAX_DEEP_TOKENS)
						contents := make([]string, 0, len(summary))
						for _, msg := range summary {
							contents = append(contents, msg.Content)
						}
						display.TypeText(fmt.Sprintf("🤖 Deep memory updated (%d messages): %v", num, contents), 0.02)
					} else {
						logging_setup.LogQueue <- "🤖 No history to remember."
					}
				} else {
					if len(chatHistory) > 0 {
						summary := memory.SummarizeKeyPoints(chatHistory, 0) // 0 для всех сообщений
						deepMemory = append(deepMemory, summary...)
						memory.SaveMemory(deepMemory, "deep_memory", config.MAX_DEEP_TOKENS)
						contents := make([]string, 0, len(summary))
						for _, msg := range summary {
							contents = append(contents, msg.Content)
						}
						display.TypeText(fmt.Sprintf("🤖 Deep memory updated: %v", contents), 0.02)
					} else {
						logging_setup.LogQueue <- "🤖 No history to remember."
					}
				}
			case strings.HasPrefix(command, "/recall") || command == "recall":
				if len(chatHistory) > 0 || len(deepMemory) > 0 || len(compiledMemory) > 0 || len(keyMemories) > 0 {
					display.TypeText("🤖 Here’s what I remember about you:", 0.02)
					if len(deepMemory) > 0 {
						contents := make([]string, 0, len(deepMemory))
						for _, msg := range deepMemory {
							contents = append(contents, msg.Content)
						}
						display.TypeText(fmt.Sprintf("From deep memory (preferences): %v", contents), "\033[36m", 0.02)
					}
					if len(keyMemories) > 0 {
						contents := make([]string, 0, len(keyMemories))
						for _, msg := range keyMemories {
							contents = append(contents, msg.Content)
						}
						display.TypeText(fmt.Sprintf("From key moments: %v", contents), "\033[36m", 0.02)
					}
					if len(compiledMemory) > 0 {
						lastFive := compiledMemory[len(compiledMemory)-5:]
						contents := make([]string, 0, len(lastFive))
						for _, msg := range lastFive {
							contents = append(contents, msg.Content)
						}
						display.TypeText(fmt.Sprintf("From compiled memory: %v", contents), "\033[36m", 0.02)
					}
					if len(chatHistory) > 0 {
						userInfo := make([]string, 0)
						for _, msg := range chatHistory {
							if msg.Role == "user" {
								userInfo = append(userInfo, msg.Content)
							}
						}
						display.TypeText(fmt.Sprintf("From current session: %v", userInfo), "\033[36m", 0.02)
					}
				} else {
					logging_setup.LogQueue <- "🤖 I don’t know anything about you yet."
				}
			}
			continue
		}

		// Анализ ввода для памяти
		memoryType := memory.AnalyzeInputForMemory(userInput)
		if memoryType == "deep" {
			deepMemory = append(deepMemory, Message{Role: "user", Content: userInput})
			memory.SaveMemory(deepMemory, "deep_memory", config.MAX_DEEP_TOKENS)
			logging_setup.LogQueue <- fmt.Sprintf("🤖 Saved to deep memory: %s", userInput)
		} else if memoryType == "key" {
			keyMemories = append(keyMemories, Message{Role: "user", Content: userInput})
			memory.SaveMemory(keyMemories, "key_memories", 0)
			logging_setup.LogQueue <- fmt.Sprintf("🤖 Saved to key memory: %s", userInput)
		}

		// Обработка промпта
		promptParts := utils.SplitText(userInput, config.MAX_PROMPT_TOKENS, "prompt")
		combinedResponse := ""

		var responseWg sync.WaitGroup
		responses := make(chan string, len(promptParts))
		for _, part := range promptParts {
			responseWg.Add(1)
			go func(p string) {
				defer responseWg.Done()
				resp, _ := ai_processing.ProcessPromptPart(p, chatHistory, keyMemories, deepMemory, compiledMemory)
				responses <- resp
			}(part)
		}
		go func() {
			responseWg.Wait()
			close(responses)
		}()

		for resp := range responses {
			combinedResponse += resp + " "
		}
		combinedResponse = strings.TrimSpace(combinedResponse)

		chatHistory = append(chatHistory, Message{Role: "user", Content: userInput})
		chatHistory = append(chatHistory, Message{Role: "assistant", Content: combinedResponse})
		messageCount++

		if messageCount >= config.COMPILE_INTERVAL {
			newCompiledBlock := memory.CompileMemory(chatHistory, keyMemories, compiledMemory)
			compiledMemory = append(compiledMemory, newCompiledBlock...)
			memory.SaveMemory(compiledMemory, "compiled_memory", 0)
			chatHistory = chatHistory[len(chatHistory)-config.MAX_HISTORY_SIZE*2:]
			messageCount = 0
			contents := make([]string, 0, len(newCompiledBlock))
			for _, msg := range newCompiledBlock {
				contents = append(contents, msg.Content)
			}
			display.TypeText(fmt.Sprintf("🤖 Memory compiled: %v", contents), 0.02)
		} else {
			if len(chatHistory) > config.MAX_HISTORY_SIZE*2 {
				chatHistory = chatHistory[len(chatHistory)-config.MAX_HISTORY_SIZE*2:]
			}
			memory.SaveMemory(chatHistory, "chat_history", config.MAX_CHAT_TOKENS)
		}
	}
}

// Вспомогательные функции для парсинга чисел
func isDigit(s string) bool {
	_, err := parseInt(s)
	return err == nil
}

func parseInt(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}