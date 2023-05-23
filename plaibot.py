from datetime import datetime
import discord
from discord.ext import commands
import logging
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("paperbot")
formatter = logging.Formatter("🗃️|%(asctime)s|%(message)s")
# Set up console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)
# Set up file handler
fh = logging.FileHandler(f'_paperbot_{datetime.now().strftime("%d%m%y")}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
log.info(f"ROOT_DIR: {ROOT_DIR}")
KEYS_DIR = os.path.join(ROOT_DIR, ".keys")
log.info(f"KEYS_DIR: {KEYS_DIR}")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
log.info(f"DATA_DIR: {DATA_DIR}")

def set_discord_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "discord.txt"), "r") as f:
                bot_token = f.read()
        except FileNotFoundError:
            log.warning("Discord API key not found.")
            return
    os.environ["DISCORD_BOT_TOKEN"] = bot_token
    log.info("Discord API key set.")

class PlaiBot(commands.Bot):

    TEST = 1110662456323342417
    PROD = 1107745177264726036

    def __init__(self, command_prefix, **options):
        super().__init__(command_prefix, **options)
        self.robot_api = None
        self.channel_id = self.TEST  

    async def on_ready(self):
        log.info(f'We have logged in as {self.user}')
        await self.get_channel(self.channel_id).send("plaibot is here!")

    async def on_message(self, message):
        if message.author == self.user:
            return
        log.debug(f"Received message: {message}")
        if message.content.startswith('$hello'):
            await self.get_channel(self.channel_id).send('Hello!')
            log.info(f'Sent hello to {message.author}')

    async def on_command(self, ctx):
        log.info(f"Received command: {ctx.message.content}")    

    @commands.command('wiggle')
    async def wiggle(self, ctx, duration: float = 5.0):
        self.robot_api.rotate(duration)
        self.robot_api.move(duration)
        await ctx.send(f'Wiggled the robot for {duration} seconds.')
        log.info(f'Wiggled the robot for {duration} seconds')

    @commands.command('capture_image')
    async def capture_image(self, ctx):
        image_path = self.robot_api.capture_image()
        await ctx.send(file=discord.File(image_path))
        log.info(f'Sent image to {ctx.message.author}')

if __name__ == "__main__":
    set_discord_key()
    intents = discord.Intents.default()
    intents.message_content = True
    bot = PlaiBot(
        command_prefix='$',
        intents=intents,
    )
    bot.run(os.environ.get('DISCORD_BOT_TOKEN'))
