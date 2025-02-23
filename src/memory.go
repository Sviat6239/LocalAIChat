package memory

import (
	"database/sql"
	"fmt"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3" // Драйвер SQLite для Go

	"yourproject/config"    // Замените на реальный путь
	"yourproject/logging_setup"
	"yourproject/utils"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Инициализация базы данных SQLite
func InitDB() {
	db, err := sql.Open("sqlite3", config.MEMORY_DB)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error opening database: %v\033[0m", err)
		return
	}
	defer db.Close()

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS chat_history (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			role TEXT,
			content TEXT
		)`)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error creating chat_history table: %v\033[0m", err)
		return
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS key_memories (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			role TEXT,
			content TEXT
		)`)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error creating key_memories table: %v\033[0m", err)
		return
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS deep_memory (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			role TEXT,
			content TEXT
		)`)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error creating deep_memory table: %v\033[0m", err)
		return
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS compiled_memory (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			role TEXT,
			content TEXT
		)`)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error creating compiled_memory table: %v\033[0m", err)
	}
}

// Вызов инициализации при загрузке пакета
func init() {
	InitDB()
}

// LoadMemory загружает память из базы данных
func LoadMemory(tableName string, defaultValue []Message) []Message {
	db, err := sql.Open("sqlite3", config.MEMORY_DB)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error opening database: %v\033[0m", err)
		return defaultValue
	}
	defer db.Close()

	rows, err := db.Query(fmt.Sprintf("SELECT role, content FROM %s", tableName))
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error querying %s: %v\033[0m", tableName, err)
		return defaultValue
	}
	defer rows.Close()

	var result []Message
	for rows.Next() {
		var role, content string
		if err := rows.Scan(&role, &content); err != nil {
			logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error scanning row from %s: %v\033[0m", tableName, err)
			continue
		}
		result = append(result, Message{Role: role, Content: content})
	}

	if err := rows.Err(); err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error iterating rows from %s: %v\033[0m", tableName, err)
		return defaultValue
	}

	if tableName == "chat_history" {
		if len(result) > config.MAX_HISTORY_SIZE {
			result = result[len(result)-config.MAX_HISTORY_SIZE:]
		}
	}
	return result
}

// SaveMemory сохраняет память в базу данных
func SaveMemory(memory []Message, tableName string, maxSize int) {
	db, err := sql.Open("sqlite3", config.MEMORY_DB)
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error opening database: %v\033[0m", err)
		return
	}
	defer db.Close()

	tx, err := db.Begin()
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error starting transaction for %s: %v\033[0m", tableName, err)
		return
	}

	_, err = tx.Exec(fmt.Sprintf("DELETE FROM %s", tableName))
	if err != nil {
		tx.Rollback()
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error deleting from %s: %v\033[0m", tableName, err)
		return
	}

	if maxSize > 0 && tableName != "compiled_memory" {
		totalTokens := 0
		for i := len(memory) - 1; i >= 0; i-- {
			tokens := utils.EstimateTokens(memory[i].Content)
			if totalTokens+tokens > maxSize && i < len(memory)-1 {
				memory = memory[i+1:]
				break
			}
			totalTokens += tokens
		}
		if tableName == "chat_history" && len(memory) > config.MAX_HISTORY_SIZE {
			memory = memory[len(memory)-config.MAX_HISTORY_SIZE:]
		}
	}

	stmt, err := tx.Prepare(fmt.Sprintf("INSERT INTO %s (role, content) VALUES (?, ?)", tableName))
	if err != nil {
		tx.Rollback()
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error preparing statement for %s: %v\033[0m", tableName, err)
		return
	}

	for _, msg := range memory {
		_, err := stmt.Exec(msg.Role, msg.Content)
		if err != nil {
			tx.Rollback()
			logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error inserting into %s: %v\033[0m", tableName, err)
			return
		}
	}

	err = tx.Commit()
	if err != nil {
		logging_setup.LogQueue <- fmt.Sprintf("\033[31m⚠️ Error committing transaction for %s: %v\033[0m", tableName, err)
	}
}

// AnalyzeInputForMemory определяет, куда сохранить ввод пользователя
func AnalyzeInputForMemory(text string) string {
	textLower := strings.ToLower(text)
	deepKeywords := []string{"люблю", "хочу", "зовут", "мой", "мне", "предпочитаю", "обожаю", "тащусь", "love", "want", "called", "my", "i", "prefer"}
	keyKeywords := []string{"сделал", "произошло", "важно", "событие", "did", "happened", "important", "event"}

	if strings.HasSuffix(textLower, "?") || containsAny(textLower, []string{"что", "как", "кто", "где", "когда", "почему", "what", "how", "who", "where", "when", "why"}) {
		return ""
	}

	for _, keyword := range deepKeywords {
		if strings.Contains(textLower, keyword) {
			return "deep"
		}
	}
	for _, keyword := range keyKeywords {
		if strings.Contains(textLower, keyword) {
			return "key"
		}
	}
	if strings.Contains(textLower, "я") && containsAny(textLower, []string{"зовут", "called", "программист", "programmer"}) {
		return "deep"
	}
	return ""
}

// FilterRedundantInfo фильтрует избыточные данные
func FilterRedundantInfo(infoList []string) []string {
	filtered := make([]string, 0)
	seenConcepts := make(map[string]bool)
	for _, item := range infoList {
		keyParts := strings.Fields(item)
		key := strings.Join(keyParts, " ")
		if !seenConcepts[key] {
			filtered = append(filtered, item)
			seenConcepts[key] = true
		}
	}
	return filtered
}

// SummarizeKeyPoints суммирует последние сообщения пользователя
func SummarizeKeyPoints(chatHistory []Message, numMessages int) []Message {
	userMessages := make([]Message, 0)
	for _, msg := range chatHistory {
		if msg.Role == "user" {
			userMessages = append(userMessages, msg)
		}
	}
	if len(userMessages) > numMessages {
		userMessages = userMessages[len(userMessages)-numMessages:]
	}
	return userMessages
}

// CompileMemory компилирует память с проверкой дубликатов
func CompileMemory(chatHistory, keyMemories, previousCompiledMemory []Message) []Message {
	compiled := make([]Message, 0)
	existingContents := make(map[string]bool)
	for _, msg := range previousCompiledMemory {
		existingContents[msg.Content] = true
	}

	keywordCount := make(map[string]int)
	for _, msg := range chatHistory {
		if msg.Role == "user" {
			compiledContent := fmt.Sprintf("User said: %s", msg.Content)
			if !existingContents[compiledContent] {
				compiled = append(compiled, Message{Role: "compiled", Content: compiledContent})
				words := strings.Fields(strings.ToLower(msg.Content))
				for _, word := range words {
					if word == "love" || word == "want" || word == "called" {
						keywordCount[word]++
					}
				}
			}
		}
	}

	for _, msg := range keyMemories {
		if !existingContents[msg.Content] {
			compiled = append(compiled, msg)
		}
	}

	finalCompiled := make([]Message, 0)
	seenKeywords := make(map[string]bool)
	for _, entry := range compiled {
		contentLower := strings.ToLower(entry.Content)
		keyPhrase := ""
		words := strings.Fields(contentLower)
		for _, word := range words {
			if word == "love" || word == "want" || word == "called" {
				keyPhrase += word + " "
			}
		}
		keyPhrase = strings.TrimSpace(keyPhrase)
		if keyPhrase == "" || !seenKeywords[keyPhrase] {
			finalCompiled = append(finalCompiled, entry)
			seenKeywords[keyPhrase] = true
		}
	}

	return finalCompiled
}

// Кастомный кэш с TTL 10 минут (600 секунд)
type TTLCache struct {
	items     map[interface{}]struct {
		value interface{}
		exp   time.Time
	}
	mu       sync.Mutex
	maxSize  int
	ttl      time.Duration
}

func NewTTLCache(maxSize int, ttl time.Duration) *TTLCache {
	return &TTLCache{
		items:    make(map[interface{}]struct {
			value interface{}
			exp   time.Time
		}),
		maxSize: maxSize,
		ttl:     ttl,
	}
}

func (c *TTLCache) Get(key interface{}) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if item, exists := c.items[key]; exists {
		if time.Now().Before(item.exp) {
			return item.value, true
		}
		delete(c.items, key)
	}
	return nil, false
}

func (c *TTLCache) Set(key, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.items) >= c.maxSize {
		// Удаляем самый старый элемент (простая эмуляция, можно улучшить с использованием списка)
		for k := range c.items {
			delete(c.items, k)
			break
		}
	}
	c.items[key] = struct {
		value interface{}
		exp   time.Time
	}{value: value, exp: time.Now().Add(c.ttl)}
}

// SplitContext разделяет контекст на части по токенам
func SplitContext(chatHistory []Message, userInputPart string, keyMemories, deepMemory, compiledMemory []Message, maxTokens int) ([][]Message, bool, error) {
	cacheKey := createCacheKey(chatHistory, userInputPart, keyMemories, deepMemory, compiledMemory)
	if value, exists := contextCache.Get(cacheKey); exists {
		return value.([][]Message), false, nil
	}

	contextParts := []string{config.DEFAULT_SYSTEM_PROMPT.Content}
	if config.CUSTOM_INSTRUCTIONS != "" {
		contextParts = append(contextParts, fmt.Sprintf("Custom Instructions: %s", config.CUSTOM_INSTRUCTIONS))
	}
	if len(deepMemory) > 0 {
		contextParts = append(contextParts, "Deep Memory (preferences):")
		for _, msg := range deepMemory {
			contextParts = append(contextParts, fmt.Sprintf("%s: %s", msg.Role, msg.Content))
		}
	}
	if len(keyMemories) > 0 {
		contextParts = append(contextParts, "Key Moments:")
		for _, msg := range keyMemories {
			contextParts = append(contextParts, fmt.Sprintf("%s: %s", msg.Role, msg.Content))
		}
	}
	if len(compiledMemory) > 0 {
		contextParts = append(contextParts, "Compiled Memory:")
		for _, msg := range compiledMemory {
			contextParts = append(contextParts, fmt.Sprintf("%s: %s", msg.Role, msg.Content))
		}
	}
	contextParts = append(contextParts, "Current Conversation:")
	for _, msg := range chatHistory[len(chatHistory)-config.MAX_HISTORY_SIZE:] {
		contextParts = append(contextParts, fmt.Sprintf("%s: %s", msg.Role, msg.Content))
	}
	contextParts = append(contextParts, fmt.Sprintf("user: %s", userInputPart))

	contextParts = utils.CompressContext(contextParts)

	totalTokens := utils.EstimateTokens(strings.Join(contextParts, "\n"))
	if totalTokens <= maxTokens {
		result := [][]Message{
			append([]Message{config.DEFAULT_SYSTEM_PROMPT},
				append([]Message{},
					append(deepMemory,
						append(keyMemories,
							append(compiledMemory,
								append(chatHistory[len(chatHistory)-config.MAX_HISTORY_SIZE:],
									Message{Role: "user", Content: userInputPart},
								)...,
							)...,
						)...,
					)...,
				),
			},
		}
		contextCache.Set(cacheKey, result)
		return result, false, nil
	}

	parts := make([][]Message, 0)
	currentPart := []Message{config.DEFAULT_SYSTEM_PROMPT}
	if config.CUSTOM_INSTRUCTIONS != "" {
		currentPart = append(currentPart, Message{Role: "system", Content: fmt.Sprintf("Custom Instructions: %s", config.CUSTOM_INSTRUCTIONS)})
	}
	currentPart = append(currentPart, deepMemory...)
	currentTokens := utils.EstimateTokens(config.DEFAULT_SYSTEM_PROMPT.Content)
	if config.CUSTOM_INSTRUCTIONS != "" {
		currentTokens += utils.EstimateTokens(config.CUSTOM_INSTRUCTIONS)
	}
	for _, msg := range deepMemory {
		currentTokens += utils.EstimateTokens(msg.Content)
	}

	allMessages := append(append(keyMemories, compiledMemory...), chatHistory[len(chatHistory)-config.MAX_HISTORY_SIZE:]...)
	allMessages = append(allMessages, Message{Role: "user", Content: userInputPart})
	splitOccurred := false

	for _, msg := range allMessages {
		msgTokens := utils.EstimateTokens(msg.Content)
		if currentTokens+msgTokens > maxTokens && len(currentPart) > len([]Message{config.DEFAULT_SYSTEM_PROMPT})+(1 if config.CUSTOM_INSTRUCTIONS != "" else 0)+len(deepMemory) {
			parts = append(parts, currentPart)
			currentPart = []Message{config.DEFAULT_SYSTEM_PROMPT}
			if config.CUSTOM_INSTRUCTIONS != "" {
				currentPart = append(currentPart, Message{Role: "system", Content: fmt.Sprintf("Custom Instructions: %s", config.CUSTOM_INSTRUCTIONS)})
			}
			currentPart = append(currentPart, deepMemory...)
			currentPart = append(currentPart, msg)
			currentTokens = utils.EstimateTokens(config.DEFAULT_SYSTEM_PROMPT.Content)
			if config.CUSTOM_INSTRUCTIONS != "" {
				currentTokens += utils.EstimateTokens(config.CUSTOM_INSTRUCTIONS)
			}
			for _, m := range deepMemory {
				currentTokens += utils.EstimateTokens(m.Content)
			}
			currentTokens += msgTokens
			splitOccurred = true
		} else {
			currentPart = append(currentPart, msg)
			currentTokens += msgTokens
		}
	}

	if len(currentPart) > 0 {
		parts = append(parts, currentPart)
	}

	contextCache.Set(cacheKey, parts)
	return parts, splitOccurred, nil
}

// Вспомогательная функция для создания ключа кэша
func createCacheKey(chatHistory []Message, userInputPart string, keyMemories, deepMemory, compiledMemory []Message) interface{} {
	var keyParts []string
	for _, msg := range chatHistory {
		keyParts = append(keyParts, fmt.Sprintf("%v", msg))
	}
	keyParts = append(keyParts, userInputPart)
	for _, msg := range keyMemories {
		keyParts = append(keyParts, fmt.Sprintf("%v", msg))
	}
	for _, msg := range deepMemory {
		keyParts = append(keyParts, fmt.Sprintf("%v", msg))
	}
	for _, msg := range compiledMemory {
		keyParts = append(keyParts, fmt.Sprintf("%v", msg))
	}
	return strings.Join(keyParts, "|")
}

// Глобальный кэш для контекста с TTL 10 минут (600 секунд)
var contextCache = NewTTLCache(100, 10*time.Minute)

// Вспомогательная функция для проверки наличия любого из подстрок
func containsAny(text string, substrings []string) bool {
	for _, sub := range substrings {
		if strings.Contains(text, sub) {
			return true
		}
	}
	return false
}