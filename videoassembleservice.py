import os
# os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
# Нужно указать путь до cuda в некоторых случаях установки opencv+cuda
import cv2
import asyncpg
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


# Подключение к PostgreSQL
async def init_db():
    return await asyncpg.create_pool(dsn='postgresql://username:password@localhost:5432')


# Создание пула соединений с базой данных
db_pool = None  # Глобальная переменная для пула подключений


class ProcessVideoRequest(BaseModel):
    id: int
    class_numbers: str  # class_numbers как строка
    fps: int


# Функция для получения информации о запросе из базы данных
async def get_request_info(request_id):
    async with db_pool.acquire() as connection:
        row = await connection.fetchrow('SELECT * FROM requests WHERE id = $1', request_id)
        return dict(row) if row else None


# Функция для получения детекций по кадру
async def get_detections(request_id, frame_id):
    async with db_pool.acquire() as connection:
        rows = await connection.fetch('SELECT * FROM detections WHERE request_id = $1 AND frame_id = $2', request_id, frame_id)
        return [dict(row) for row in rows]


# Асинхронная функция для обновления статуса запроса
async def update_request_status(request_id, status):
    async with db_pool.acquire() as connection:
        await connection.execute('UPDATE requests SET status = $1 WHERE id = $2', status, request_id)


# Обработка видео для сборки из кадров
@app.post('/assemble_video')
async def assemble_video(request: ProcessVideoRequest):
    global db_pool

    request_id = request.id
    class_numbers_str = request.class_numbers
    fps_res = request.fps

    try:
        if db_pool is None:
            db_pool = await init_db()

        request_info = await get_request_info(request_id)
        if not request_info:
            raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found.")

        frames_path = f'frames/{request_id}'
        if not os.path.exists(frames_path):
            raise HTTPException(status_code=404, detail=f"Frames for Request ID {request_id} not found.")

        frame_files = sorted(os.listdir(frames_path))
        if not frame_files:
            raise HTTPException(status_code=404, detail=f"No frames found for Request ID {request_id}.")

        # Преобразование строки class_numbers в список целых чисел, игнорируя пустые значения
        class_numbers = [int(num) for num in class_numbers_str.split(",") if num.isdigit()]

        # Получаем размер первого кадра для настройки видео
        first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
        height, width, layers = first_frame.shape
        video_path = os.path.join(frames_path, f'{request_id}_result.avi')

        # Настраиваем видео писатель
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, float(fps_res), (width, height))

        for frame_file in frame_files:
            frame_id = int(os.path.splitext(frame_file)[0])
            frame_path = os.path.join(frames_path, frame_file)
            frame = cv2.imread(frame_path)

            detections = await get_detections(request_id, frame_id)
            filtered_detections = [d for d in detections if d['class'] in class_numbers]

            for detection in filtered_detections:
                x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out.write(frame)

        out.release()
        print(f'video {request_id} assembled')
        await update_request_status(request_id, 'video_complete')
        return {"status": "video_complete", "request_id": request_id, "video_path": video_path}

    except Exception as e:
        print(f"Error assembling video for request_id {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5567)
