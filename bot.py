import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import BufferedInputFile
from PIL import Image
import io
from pathlib import Path
import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as F

from config_reader import config
from resnet_generator import *

logging.basicConfig(level=logging.INFO)
bot = Bot(token=config.bot_token.get_secret_value())
dp = Dispatcher()


def generate(image):
    transforms = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    gen = torch.load('generator.pkl')
    if image.width > 512 or image.height > 512:
        image = F.resize(image, 512)
    tensor = transforms(image).unsqueeze(0)
    with torch.no_grad():
        tensor = gen(tensor).squeeze()
    return Image.fromarray(((tensor.detach().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype('uint8'))


@dp.message(commands=["start"])
async def cmd_start(message: types.Message):
    await message.answer("Hello!")


@dp.message(content_types="photo")
async def process_photo(message: types.Message, bot: Bot):
    photo_id = message.photo[-1].file_id
    photo_file = await bot.get_file(photo_id)
    photo_path = Path(photo_file.file_path)
    image_bytes: io.BytesIO = await bot.download_file(photo_file.file_path)
    image = Image.open(image_bytes)
    image = generate(image)

    res_image_bytes = io.BytesIO()
    image.save(res_image_bytes, format='JPEG')
    image_file = BufferedInputFile(res_image_bytes.getvalue(), str(photo_path.with_suffix('.jpg')))
    await message.reply_photo(image_file)


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
