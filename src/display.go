package display

import (
	"fmt"
	"strings"
	"time"

	"yourproject/config" // Замените на реальный путь
)

func TypeText(text string, color string, delay float64) {
	// Используем ANSI-коды для цветов (зелёный по умолчанию, как в colorama.Fore.GREEN)
	if color == "" {
		color = "\033[32m" // Зелёный цвет
	}
	reset := "\033[0m" // Сброс цвета

	for _, char := range text {
		fmt.Print(color + string(char))
		time.Sleep(time.Duration(delay * 1000 * 1000 * 1000)) // Задержка в наносекундах
	}
	fmt.Print(reset)
}

func DisplayToken(token string, inCodeBlock bool, lastTokenWasSpace bool) bool {
	// Проверка на пустой токен вне блока кода
	if !strings.TrimSpace(token) != "" && !inCodeBlock {
		return lastTokenWasSpace
	}

	// Обработка блоков кода (```)
	if strings.HasPrefix(token, "```") || strings.HasSuffix(token, "```") {
		if inCodeBlock && strings.HasSuffix(token, "```") {
			fmt.Print("\033[32m" + token + "\n") // Зеленый цвет для кода
			time.Sleep(time.Duration(config.TOKEN_DISPLAY_DELAY * 1000 * 1000 * 1000))
			return false
		} else if !inCodeBlock && strings.HasPrefix(token, "```") {
			fmt.Print("\033[32m" + token + "\n") // Зеленый цвет для кода
			time.Sleep(time.Duration(config.TOKEN_DISPLAY_DELAY * 1000 * 1000 * 1000))
			return true
		}
		return inCodeBlock
	}

	// Отображение токена внутри блока кода
	if inCodeBlock {
		fmt.Print("\033[32m" + token) // Зеленый цвет для кода
		time.Sleep(time.Duration(config.TOKEN_DISPLAY_DELAY * 1000 * 1000 * 1000))
		return strings.HasSuffix(token, "\n") || strings.HasSuffix(token, " ")
	}

	// Отображение токена вне блока кода
	if lastTokenWasSpace && !strings.HasPrefix(token, ".") && !strings.HasPrefix(token, ",") && !strings.HasPrefix(token, "!") &&
		!strings.HasPrefix(token, "?") && !strings.HasPrefix(token, ":") && !strings.HasPrefix(token, ";") && !strings.HasPrefix(token, "\n") &&
		!strings.HasPrefix(token, " ") {
		fmt.Print("\033[32m ")
	}
	fmt.Print("\033[32m" + token)
	time.Sleep(time.Duration(config.TOKEN_DISPLAY_DELAY * 1000 * 1000 * 1000))
	return strings.HasSuffix(token, " ") || strings.HasSuffix(token, "\n")
}