import logging
import os
import re
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Union
import polars as pd

import arxiv
import discord
import google.generativeai as palm
import openai

EMOJI = "🗃️"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_DIR = os.path.join(ROOT_DIR, ".keys")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_DIR = os.path.join(ROOT_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("paperbot")
formatter = logging.Formatter(f"{EMOJI}" + "|%(asctime)s|%(message)s")
# Set up console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)
# Set up file handler
logfile_name = f'_paperbot_{datetime.now().strftime("%d%m%y")}.log'
logfile_path = os.path.join(LOG_DIR, logfile_name)
fh = logging.FileHandler(logfile_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

log.info(f"ROOT_DIR: {ROOT_DIR}")
log.info(f"KEYS_DIR: {KEYS_DIR}")
log.info(f"DATA_DIR: {DATA_DIR}")
log.info(f"LOG_DIR: {LOG_DIR}")


def set_openai_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "openai.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("OpenAI API key not found.")
            return
    os.environ["OPENAI_API_KEY"] = key
    openai.api_key = key
    log.info("OpenAI API key set.")


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


def set_huggingface_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "huggingface.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("HuggingFace API key not found.")
            return
    os.environ["HUGGINGFACE_API_KEY"] = key
    log.info("HuggingFace API key set.")


def set_palm_key(key=None):
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "palm.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Palm API key not found.")
            return
    os.environ["PALM_API_KEY"] = key
    # palm.configure(api_key=key)
    log.info("Palm API key set.")


def palm_text(prompt):
    """https://developers.generativeai.google/tutorials/text_quickstart"""

    models = [
        m
        for m in palm.list_models()
        if "generateText" in m.supported_generation_methods
    ]
    model = models[0].name
    print(model)

    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        # The maximum length of the response
        max_output_tokens=800,
    )

    return completion.result


def find_papers(msg: str) -> Iterator[arxiv.Result]:
    pattern = r"arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)"
    matches = re.findall(pattern, msg)
    for arxiv_id in matches:
        search = arxiv.Search(id_list=[arxiv_id])
        for paper in search.results():
            yield paper


def gpt_text(
    prompt: Union[str, List[Dict[str, str]]] = None,
    system=None,
    model="gpt-3.5-turbo",
    temperature: float = 0.6,
    max_tokens: int = 32,
    stop: List[str] = ["\n"],
):
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif prompt is None:
        prompt = []
    if system is not None:
        prompt = [{"role": "system", "content": system}] + prompt
    log.debug(f"Function call to GPT {model}: \n {prompt}")
    response: Dict = openai.ChatCompletion.create(
        messages=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    return response["choices"][0]["message"]["content"]


class LocalDB(object):

    COLUMNS = ["title", "url", "authors", "published"]

    def __init__(self,
                 filepath: str = None,
        ):
        if filepath is None:
            log.info("No filepath provided for local DB.")
            filepath = os.path.join(DATA_DIR, "papers.csv")
        if os.path.exists(filepath):
            log.info(f"Loading local DB from {filepath}")
            self.df = pd.read_csv(filepath)
        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)

    def add_paper(self, paper: arxiv.Result):
        self.df = self.df.append(
            {
                "title": paper.title,
                "url": paper.pdf_url,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%m/%d/%Y"),
            },
            ignore_index=True,
        )

    def get_papers(self, id: str):
        return self.df[self.df["id"] == id]


class PaperBot(discord.Client):
    """
    https://github.com/Rapptz/discord.py/tree/master/examples


    actions:
    - check message for arxiv link, add to db
    - manage a priority queue, queue based on votes?
    - suggest sources for finding papers (auto suggest papers?)

    """

    # Channel IDs
    CHANNELS: Dict[str, int] = {
        "papers": 1107745177264726036,
        "bot-debug": 1110662456323342417,
    }

    # Paper sources
    SOURCES = """
[Twitter](https://twitter.com/i/lists/1653485531546767361)
[PapersWithCode](https://paperswithcode.com/)
[Reddit](https://www.reddit.com/user/deephugs/m/ml/)
[ArxivSanity](http://www.arxiv-sanity.com/)
[LabML](https://papers.labml.ai/papers/weekly/)
"""

    def __init__(self, *args, channel_name: str = "bot-debug", **kwargs):
        super().__init__(*args, **kwargs)
        self.db = LocalDB()
        if self.CHANNELS.get(channel_name, None) is None:
            raise ValueError(f"Channel {channel_name} not found.")
        self.channel_id: int = self.CHANNELS[channel_name]
        self.actions: Dict[str, Callable] = {
            "add_paper": self.add_paper,
            "paper_sources": self.paper_sources,
            "chat": self.chat,
            "image": self.capture_image,
        }
        self.action_list: List[str] = list(self.actions.keys())

    async def on_ready(self):
        log.info(f"We have logged in as {self.user}")
        await self.get_channel(self.channel_id).send(f"{EMOJI} has entered the chat!")

    async def on_message(self, msg: discord.Message):
        if msg.author.id == self.user.id:
            return
        log.debug(f"Received message: {msg.content}")
        if self.user.id in [m.id for m in msg.mentions]:
            log.debug(f"Mentioned in message by {msg.author.name}")
            behavior = gpt_text(
                prompt=f"{msg.content}",
                system=" ".join(
                    [
                        "You are paperbot."
                        "Determine which behavior the user wants to run.",
                        "Do not explain, Return the name of the behavior only.",
                        f"The available behaviors are {', '.join(self.action_list)}",
                    ]
                ),
            )
            behavior = self.actions.get(behavior, None)
            if behavior is not None:
                log.info(f"Running behavior: {behavior}")
                await self.actions[behavior](msg)

    async def add_paper(self, msg: discord.Message):
        for paper in find_papers(msg.content):
            log.info(f"Found paper: {paper.title}")
            id = paper.get_short_id()
            paper = self.db.get_papers(id)
            if paper is None:
                paper = self.db.add_paper(paper)
                _msg = f"Adding paper {id}"
                log.info(_msg)
                await self.get_channel(self.channel_id).send(_msg)
            else:
                _msg = f"The paper {id} was already posted"
                log.info(_msg)
                await self.get_channel(self.channel_id).send(_msg)

    async def paper_sources(self, msg: discord.Message):
        await self.get_channel(self.channel_id).send(self.SOURCES)

    async def chat(self, msg: discord.Message):
        pass

    async def capture_image(self, ctx):
        pass


if __name__ == "__main__":
    set_discord_key()
    set_huggingface_key()
    set_openai_key()
    set_palm_key()

    intents = discord.Intents.default()
    intents.message_content = True
    bot = PaperBot(intents=intents)
    bot.run(os.environ["DISCORD_BOT_TOKEN"])
