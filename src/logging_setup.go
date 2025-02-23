package logging_setup

import (
	"fmt"
	"sync"

	"yourproject/config" // Замените на реальный путь
)

// Канал для логов (аналог очереди Queue в Python)
var LogQueue = make(chan interface{}, 100) // Буферизированный канал на 100 сообщений

// LogWriter выполняет бесконечный цикл чтения из канала и вывода логов
func LogWriter(wg *sync.WaitGroup) {
	defer wg.Done()
	for message := range LogQueue {
		if message == nil { // Сигнал завершения
			break
		}
		// Выводим сообщение в терминал с цветами (используем ANSI-коды, как в colorama)
		fmt.Print(message.(string) + "\033[0m") // Сбрасываем цвет после вывода
	}
}

// StartLogger запускает логгер в фоновом режиме
func StartLogger() {
	var wg sync.WaitGroup
	wg.Add(1)
	go LogWriter(&wg) // Запускаем горутину для записи логов
	// Ждать завершения не нужно, так как логгер работает в фоновом режиме
}