import os

# os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
# Нужно указать путь до cuda в некоторых случаях установки opencv+cuda
import json
from fastapi import FastAPI
from aio_pika import connect_robust, IncomingMessage
from ultralytics import YOLO
import asyncpg
import aiohttp

app = FastAPI()

queue_name = "yolo_frames"
yolo_model = YOLO("yolov8m.pt").to("cuda")  # Укажите путь к вашей модели YOLOv8 и используйте CUDA
db_pool = None  # Глобальная переменная для пула подключений


# Подключение к PostgreSQL
async def init_db():
    global db_pool
    db_pool = await asyncpg.create_pool(dsn='postgresql://postgres:vanilinn@localhost:5432')


# Функция для обработки сообщений из очереди
async def on_message(message: IncomingMessage):
    async with message.process():
        try:
            body = json.loads(message.body)
            frame_path = body["frame_path"]
            request_id = body["request_id"]
            frame_number = body["frame_number"]
            is_last_frame = body["is_last_frame"]
            class_numbers = body["class_numbers"]
            fps = body["fps"]

            print(f"Received message: {body}")

            # Обработка кадра с помощью YOLOv8 на CUDA
            results = yolo_model(frame_path, device="cuda")
            detections = results[0].tojson()
            print(detections)

            # Преобразование результатов в список словарей
            detections = json.loads(detections)

            # Сохранение результатов в базе данных
            await save_detections(request_id, frame_number, detections)

            # Если это последний кадр, отправить запрос на сборку видео
            if is_last_frame:
                await send_post_request("http://127.0.0.1:5567/assemble_video", {
                    "id": request_id,
                    "class_numbers": ",".join(map(str, class_numbers)),
                    "fps": fps
                })

        except Exception as e:
            print(f"Error processing message: {e}")


# Функция для сохранения детекций в базе данных
async def save_detections(request_id, frame_id, detections):
    async with db_pool.acquire() as connection:
        for detection in detections:
            class_id = detection['class']
            confidence = detection['confidence']
            x1 = detection['box']['x1']
            y1 = detection['box']['y1']
            x2 = detection['box']['x2']
            y2 = detection['box']['y2']
            await connection.execute(
                '''
                INSERT INTO detections (request_id, frame_id, class, confidence, x1, y1, x2, y2)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ''',
                request_id, frame_id, class_id, confidence, x1, y1, x2, y2
            )


# Функция для отправки POST запроса
async def send_post_request(api_url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json=data) as response:
            if response.status != 200:
                print(f"Failed to send POST request: {response.status}")
            else:
                print(f"POST request sent successfully: {await response.json()}")


@app.on_event("startup")
async def startup_event():
    await init_db()
    try:
        connection = await connect_robust('amqp://events:events@queue.saferegion.net/')
        channel = await connection.channel()
        queue = await channel.declare_queue(queue_name, durable=True)
        await queue.consume(on_message)
        print(f"Connected to RabbitMQ queue: {queue_name}")
    except Exception as e:
        print(f"Error connecting to RabbitMQ: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5556)
