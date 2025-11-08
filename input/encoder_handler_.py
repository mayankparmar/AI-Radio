import asyncio
import random

class EncoderHandler:
    async def listen(self, callback):
        while True:
            await asyncio.sleep(5)  # Simulated encoder input
            await callback("encoder1", random.randint(-5, 5))
