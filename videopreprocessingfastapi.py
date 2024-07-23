import os

# os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
# Нужно указать путь до cuda в некоторых случаях установки opencv+cuda
import uvicorn
import cv2
import asyncpg
from aiofiles import open as aio_open
from aio_pika import connect_robust, Message, DeliveryMode
import json
from fastapi import FastAPI, HTTPException

app = FastAPI()


# Подключение к PostgreSQL
async def init_db():
    return await asyncpg.create_pool(dsn='postgresql://username:password@host:5432')


# Создание пула соединений с базой данных
db_pool = None  # Глобальная переменная для пула подключений


# Асинхронная функция для сохранения кадра
async def save_frame(frame, frame_path):
    is_success, buffer = cv2.imencode(".jpg", frame)
    async with aio_open(frame_path, mode='wb') as f:
        await f.write(buffer)


# Асинхронная функция для отправки кадра и метаданных в очередь
async def send_to_queue(frame_path, frame_metadata):
    connection = await connect_robust('amqp://username:password@host/')
    async with connection:
        channel = await connection.channel()
        await channel.declare_queue('yolo_frames', durable=True)
        message_body = json.dumps(frame_metadata).encode()
        message = Message(
            message_body,
            delivery_mode=DeliveryMode.PERSISTENT
        )
        await channel.default_exchange.publish(message, routing_key='yolo_frames')
        print(f"Sent frame metadata to queue: {frame_metadata}")


# Асинхронная функция для обновления статуса запроса
async def update_request_status(request_id, status):
    async with db_pool.acquire() as connection:
        await connection.execute('UPDATE requests SET status = $1 WHERE id = $2', status, request_id)


# Обработка видео
async def process_video(video_path, request_id, metadata, class_numbers):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"Could not open video file {video_path}")

    # Получение FPS исходного видео
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise HTTPException(status_code=500, detail="Failed to get FPS from video")

    frame_count = 0
    output_dir = f'frames/{request_id}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    request_info = await get_request_info(request_id)
    if not request_info:
        raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found.")

    fps_from_db = request_info.get('fps', 30)  # По умолчанию 30 FPS, если в базе нет значения

    # Рассчитываем интервал для сохранения кадров
    interval = int(round(source_fps / fps_from_db))

    last_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    last_saved_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_filename = f"{frame_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            is_last_frame = (frame_count + interval > last_frame_number) or (frame_count == last_frame_number)

            try:
                await save_frame(frame, frame_path)

                frame_metadata = {
                    "description": "Frame extracted from video",
                    "object_detection_task": True,
                    "source": video_path,
                    "request_id": request_id,
                    "frame_number": frame_count,
                    "frame_path": frame_path,
                    "class_numbers": class_numbers,  # Добавляем номера классов в метаданные
                    "is_last_frame": is_last_frame  # Добавляем поле is_last_frame
                }

                frame_metadata.update(metadata)

                await send_to_queue(frame_path, frame_metadata)

                if is_last_frame:
                    last_saved_frame = frame_metadata

            except Exception as e:
                print(f"Error processing frame {frame_count} for request_id {request_id}: {e}")

        frame_count += 1

    cap.release()
    await update_request_status(request_id, 'completed')
    print(f"Processed {frame_count} frames from video {video_path}.")

    if last_saved_frame and not last_saved_frame["is_last_frame"]:
        last_saved_frame["is_last_frame"] = True
        await send_to_queue(last_saved_frame["frame_path"], last_saved_frame)
        print(f"Updated last frame metadata with is_last_frame=True: {last_saved_frame['frame_number']}")


# Обработчик HTTP запросов для обработки видео
@app.post('/process_video')
async def process_video_request(request_data: dict):
    global db_pool

    request_id = request_data.get('request_id')

    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required")

    try:
        if db_pool is None:
            db_pool = await init_db()

        request_info = await get_request_info(request_id)
        if not request_info:
            raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found.")

        video_path = request_info['file_path']
        metadata = {
            "type": request_info['type'],
            "task_type": request_info['task_type'],
            "output_type": request_info['output_type'],
            "fps": request_info['fps']
        }
        class_numbers = request_info['class_numbers']  # Получаем номера классов

        await handle_video(request_id, class_numbers)
    except Exception as e:
        print(f"Error processing video request_id {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

    return {"status": "done", "request_id": request_id}


# Функция для обработки видео в фоновом режиме
async def handle_video(request_id, class_numbers):
    request_info = await get_request_info(request_id)
    if not request_info:
        print(f"Request ID {request_id} not found.")
        return

    video_path = request_info['file_path']
    metadata = {
        "type": request_info['type'],
        "task_type": request_info['task_type'],
        "output_type": request_info['output_type'],
        "fps": request_info['fps']
    }

    await process_video(video_path, request_id, metadata, class_numbers)


# Функция для получения информации о запросе из базы данных
async def get_request_info(request_id):
    async with db_pool.acquire() as connection:
        row = await connection.fetchrow('SELECT * FROM requests WHERE id = $1', request_id)
        return dict(row) if row else None


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5544)
