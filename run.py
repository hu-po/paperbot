import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Union
import polars as pd
from polars.exceptions import NoRowsReturnedError

import arxiv
import discord
import google.generativeai as palm
import openai

EMOJI: str = "🗃️"
IAM: str = "You are paperbot. You know about ML, AI, CS. You are good at explaining and suggesting literature."
SOURCES: Dict[str, str] = {
    "Twitter": "https://twitter.com/i/lists/1653485531546767361",
    "PapersWithCode": "https://paperswithcode.com/",
    "Reddit": "https://www.reddit.com/user/deephugs/m/ml/",
    "ArxivSanity": "http://www.arxiv-sanity.com/",
    "LabML": "https://papers.labml.ai/papers/weekly/",
}

MAX_TOKENS = 64
TEMPERATURE = 0

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
        max_output_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return completion.result


def find_papers(msg: str) -> Iterator[arxiv.Result]:
    pattern = r"arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)"
    matches = re.findall(pattern, msg)
    for arxiv_id in matches:
        for paper in arxiv.Search(id_list=[arxiv_id]).results():
            yield paper


def gpt_text(
    prompt: Union[str, List[Dict[str, str]]] = None,
    system=None,
    model="gpt-3.5-turbo",
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
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
    SCHEMA: Dict[str, pd.DataType] = {
        "id": pd.Utf8,
        "title": pd.Utf8,
        "url": pd.Utf8,
        "authors": pd.Utf8,
        "published": pd.Utf8,
        "abstract": pd.Utf8,
        "tags": pd.Utf8,
        "suggester": pd.Utf8,
    }

    def __init__(
        self,
        filepath: str = None,
    ):
        if filepath is None:
            log.info("No filepath provided for local DB.")
            filepath = os.path.join(DATA_DIR, "papers.csv")
        self.filepath = filepath
        if os.path.exists(filepath):
            log.info(f"Loading existing local DB from {self.filepath}")
            self.df = pd.read_csv(self.filepath)
        else:
            log.info(f"Creating new local DB at {self.filepath}")
            self.df = pd.DataFrame(schema=self.SCHEMA)

    def add_paper(self, paper: arxiv.Result):
        tags: str = gpt_text(
            prompt=f"{paper.summary}",
            system=" ".join(
                [
                    "You are paperbot.",
                    "Create a comma separated list of tags for this paper.",
                    "Do not explain, only provide the string tags.",
                    "Here is the abstract for the paper:",
                ]
            ),
        )
        _df = pd.DataFrame(
            {
                "id": paper.get_short_id(),
                "title": paper.title,
                "url": paper.pdf_url,
                "authors": ",".join([author.name for author in paper.authors]),
                "published": paper.published.strftime("%m/%d/%Y"),
                "abstract": paper.summary,
                "tags": tags,
            }
        )
        self.df = self.df.vstack(_df)
        self.save()

    def list_papers(self) -> Dict[str, Any]:
        yield from self.df.iter_rows(named=True)

    def save(self):
        log.info(f"Saving local DB to {self.filepath}")
        self.df.write_csv(self.filepath)

    def get_papers(self, id: str):
        if len(self.df) == 0:
            return None
        try:
            match = self.df.row(by_predicate=(pd.col("id") == id))
        except NoRowsReturnedError:
            return None
        return {column: value for column, value in zip(self.df.columns, match)}


async def add_paper(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
) -> None:
    for paper in find_papers(msg.content):
        log.info(f"Found paper: {paper.title}")
        id = paper.get_short_id()
        _paper = db.get_papers(id)
        if _paper is None:
            db.add_paper(paper)
            _msg = f"Adding paper {id}"
            log.info(_msg)
            await channel.send(_msg)
        else:
            _msg = f"The paper {id} was already posted"
            log.info(_msg)
            await channel.send(_msg)


async def list_papers(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
) -> None:
    embeds = []
    for paper_dict in db.list_papers():
        embeds.append(
            discord.Embed(
                title=paper_dict["title"],
                url=paper_dict["url"],
                description=paper_dict["tags"],
            )
        )
    log.info("Listing the papers.")
    await channel.send(embeds=embeds)


async def list_sources(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
) -> None:
    embeds = []
    for label, url in SOURCES.items():
        embeds.append(
            discord.Embed(
                title=label,
                url=url,
            )
        )
    log.info("Listing the sources.")
    await channel.send(content="Here are some sources for papers:", embeds=embeds)


async def list_author(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
) -> None:
    formatted_name = gpt_text(
        prompt=f"{msg.content}",
        system=" ".join(
            [
                "You are paperbot.",
                "You extract and format the author name in a message.",
                "Do not explain, only return one name in the format First,Last.",
                "Example message: 'I am interested in the work of John Smith.'",
                "Example response: 'John,Smith'",
            ]
        ),
    )
    embeds = []
    for result in arxiv.Search(query=f"au:{formatted_name}").results():
        embeds.append(
            discord.Embed(
                title=result.title,
                url=result.pdf_url,
            )
        )
    await channel.send(
        content=f"Here are the other papers by author: {formatted_name}:", embeds=embeds[:10]
    )


async def gpt_chat(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
) -> None:
    response = gpt_text(
        prompt=f"{msg.content}",
        system=" ".join(
            [
                IAM,
                "Respond to the user's message.",
            ]
        ),
    )
    await channel.send(response)


async def palm_chat(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
) -> None:
    system = " ".join(
        [
            IAM,
            "Respond to the user's message. This is the user's message:",
        ]
    )
    response = palm_text(prompt=f"{system} {msg.content}")
    await channel.send(response)


async def capture_image(self, ctx):
    pass


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

    def __init__(self, *args, channel_name: str = "bot-debug", **kwargs):
        super().__init__(*args, **kwargs)
        self.db = LocalDB()
        if self.CHANNELS.get(channel_name, None) is None:
            raise ValueError(f"Channel {channel_name} not found.")
        self.channel_id: int = self.CHANNELS[channel_name]
        self.actions: Dict[str, Callable] = {
            "add_paper": add_paper,
            "list_papers": list_papers,
            "list_sources": list_sources,
            "list_author": list_author,
            "chat": gpt_chat,
            "gpt_chat": gpt_chat,
            "palm_chat": palm_chat,
            "image": capture_image,
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
            behavior_name: str = gpt_text(
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
            behavior = self.actions.get(behavior_name, None)
            if behavior is not None:
                log.info(f"Running behavior: {behavior_name}")
                await behavior(
                    msg,
                    self.get_channel(self.channel_id),
                    self.db,
                )


if __name__ == "__main__":
    set_discord_key()
    set_huggingface_key()
    set_openai_key()
    set_palm_key()

    intents = discord.Intents.default()
    intents.message_content = True
    bot = PaperBot(intents=intents)
    bot.run(os.environ["DISCORD_BOT_TOKEN"])
