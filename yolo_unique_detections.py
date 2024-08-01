import os

# Укажите путь до CUDA
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
import json
from fastapi import FastAPI, HTTPException
from collections import defaultdict
import numpy as np
import cv2
import asyncpg
from aiofiles import open as aio_open
from ultralytics import YOLO

app = FastAPI()


# Подключение к PostgreSQL
async def init_db():
    return await asyncpg.create_pool(dsn='postgresql://login:password@localhost:5432')


# Создание пула соединений с базой данных
db_pool = None  # Глобальная переменная для пула подключений


# Асинхронная функция для сохранения изображения объекта
async def save_object_image(image, image_path):
    is_success, buffer = cv2.imencode(".jpg", image)
    async with aio_open(image_path, mode='wb') as f:
        await f.write(buffer)


# Асинхронная функция для обновления статуса запроса
async def update_request_status(request_id, status):
    async with db_pool.acquire() as connection:
        await connection.execute('UPDATE requests SET status = $1 WHERE id = $2', status, request_id)


# Асинхронная функция для сохранения детекций уникальных объектов в базе данных
async def save_unique_detections(request_id, frame_id, track_id, box, track, class_id, image_path):
    async with db_pool.acquire() as connection:
        x, y, w, h = map(float, box)
        path = image_path  # Путь до изображения
        await connection.execute(
            '''
            INSERT INTO unique_detections (request_id, frame_id, track_id, x, y, w, h, path, class_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ''',
            request_id, frame_id, track_id, x, y, w, h, path, class_id
        )


# Обработка видео и обнаружение уникальных объектов
async def process_video_for_objects(video_path, request_id, model_path, class_numbers, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"Could not open video file {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise HTTPException(status_code=500, detail="Failed to get FPS from video")

    model = YOLO(model_path)
    track_history = defaultdict(lambda: [])

    frame_count = 0
    output_dir = f'objects/{request_id}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    interval = int(round(source_fps / fps))
    unique_tracks = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            try:
                results = model.track(frame, persist=True)
                boxes = results[0].boxes.xywh.cpu() if results[0].boxes.xywh is not None else None
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else None
                classes = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes.cls is not None else None

                if boxes is None or track_ids is None or classes is None:
                    print(f"Skipping frame {frame_count} due to missing detection data.")
                    frame_count += 1
                    continue

                print(boxes, track_ids, classes, sep='\n')

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    if cls in class_numbers and track_id not in unique_tracks:
                        unique_tracks.add(track_id)
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))

                        x, y, w, h = map(int, box)
                        x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
                        x2, y2 = min(frame.shape[1], x + w // 2), min(frame.shape[0], y + h // 2)
                        object_image = frame[y1:y2, x1:x2]
                        object_filename = f"{track_id}_{frame_count:04d}.jpg"
                        object_path = os.path.join(output_dir, object_filename)

                        try:
                            await save_object_image(object_image, object_path)
                            await save_unique_detections(request_id, frame_count, track_id, box, track, cls,
                                                         object_path)
                        except Exception as e:
                            print(
                                f"Error processing object {track_id} in frame {frame_count} for request_id {request_id}: {e}")

            except Exception as e:
                print(f"Error during object detection in frame {frame_count}: {e}")

        frame_count += 1

    cap.release()
    await update_request_status(request_id, 'completed')
    print(f"Processed video {video_path} and detected unique objects.")


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
        model_path = "yolov8m.pt"  # путь к модели YOLO
        class_numbers_str = request_info['class_numbers']  # получаем строку с номерами классов
        class_numbers = list(map(int, class_numbers_str.split(',')))  # преобразуем в список чисел
        fps = request_info.get('fps', 30)  # По умолчанию 30 FPS, если в базе нет значения

        await handle_video(request_id, model_path, class_numbers, fps)
    except Exception as e:
        print(f"Error processing video request_id {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

    return {"status": "done", "request_id": request_id}


# Функция для обработки видео в фоновом режиме
async def handle_video(request_id, model_path, class_numbers, fps):
    request_info = await get_request_info(request_id)
    if not request_info:
        print(f"Request ID {request_id} not found.")
        return

    video_path = request_info['file_path']

    await process_video_for_objects(video_path, request_id, model_path, class_numbers, fps)


# Функция для получения информации о запросе из базы данных
async def get_request_info(request_id):
    async with db_pool.acquire() as connection:
        row = await connection.fetchrow('SELECT * FROM requests WHERE id = $1', request_id)
        return dict(row) if row else None


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=5548)
