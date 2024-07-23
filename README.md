# Асинхронные Backend Сервисы для Создания Умной Аналитики с Использованием YOLO

Этот проект включает три сервиса для обработки видео, распознавания объектов с помощью YOLO и сборки обработанных видео.

## Архитектура

### 1. `videopreprocessingfastapi.py`
API для разбиения видео на изображения. Получает запрос с `id` задачи, извлекает данные из базы данных PostgreSQL, разбивает видео на кадры и сохраняет каждый кадр, отправляя сообщение в очередь RabbitMQ.

### 2. `yoloservice.py`
Асинхронный сервис, который слушает очередь RabbitMQ. Для каждого сообщения сервис обрабатывает кадр с помощью модели YOLO, сохраняет информацию об объектах в таблицу `detections` базы данных PostgreSQL и проверяет, является ли этот кадр последним для текущей задачи. Если кадр последний, сервис вызывает API для сборки видео.

### 3. `videoassembleservice.py`
API для сборки видео из обработанных кадров. В зависимости от типа задания и фильтрации объектов, отмечает нужные объекты на каждом кадре и собирает из них видео с заданным fps.

## Стек Технологий

- Python 3.x
- FastAPI
- PostgreSQL
- RabbitMQ
- YOLO (You Only Look Once)

## Установка и Запуск

### 1. Клонирование Репозитория

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
