import asyncio
import json

from fastapi import FastAPI, HTTPException
from aio_pika import connect_robust, IncomingMessage

app = FastAPI()

queue_name = "yolo_frames"
import torch
print(torch.cuda.is_available())


async def on_message(message: IncomingMessage):
    async with message.process():
        body = message.body.decode()
        print('#' * 30)
        print(f"Received message: {body}")


@app.on_event("startup")
async def startup_event():
    try:
        connection = await connect_robust('amqp://username:password@host/')
        channel = await connection.channel()
        queue = await channel.declare_queue(queue_name, durable=True)
        await queue.consume(on_message)
        print(f"Connected to RabbitMQ and consuming from queue: {queue_name}")
    except Exception as e:
        print(f"Failed to connect to RabbitMQ: {e}")


@app.get("/")
async def read_root():
    return {"status": "Service is running and consuming messages from the queue"}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=5553)


