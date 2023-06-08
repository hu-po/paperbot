import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Union

import numpy as np
import arxiv
import discord
import google.generativeai as palm
import openai
import polars as pl
from polars.exceptions import NoRowsReturnedError

NAME: str = "paperbot"
EMOJI: str = "🗃️"
IAM: str = "".join(
    [
        f"You are {NAME}, a arxiv chatbot.",
        # f"You are {NAME}, a helpful bot.",
        # "You help people organize arxiv papers.",
    ]
)
DATEFORMAT = "%d.%m.%y"
SOURCES: Dict[str, str] = {
    "Twitter": "https://twitter.com/i/lists/1653485531546767361",
    "PapersWithCode": "https://paperswithcode.com/",
    "Reddit": "https://www.reddit.com/user/deephugs/m/ml/",
    "ArxivSanity": "http://www.arxiv-sanity.com/",
    "LabML": "https://papers.labml.ai/papers/weekly/",
}

DEFAULT_LLM: str = "gpt-3.5-turbo"
assert DEFAULT_LLM in ["gpt-3.5-turbo", "gpt-4", "palm"]
DEFAULT_TEMPERATURE: float = 0
DEFAULT_MAX_TOKENS: int = 64

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
logfile_name = f'_paperbot_{datetime.now().strftime(DATEFORMAT)}.log'
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

# TODO: Decorator to time functions and add log statements

def find_papers(msg: str) -> Iterator[arxiv.Result]:
    pattern = r"arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)"
    matches = re.findall(pattern, msg)
    for arxiv_id in matches:
        for paper in arxiv.Search(id_list=[arxiv_id]).results():
            yield paper


def palm_text(
    prompt: str = "",
    system: str = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """https://developers.generativeai.google/tutorials/text_quickstart"""
    models = [
        m
        for m in palm.list_models()
        if "generateText" in m.supported_generation_methods
    ]
    assert len(models) > 0, "No models found for PaLM."
    model = models[0].name
    if system is not None:
        prompt = f"{system} {prompt}"
    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    return completion.result


def gpt_text(
    prompt: Union[str, List[Dict[str, str]]] = None,
    system=None,
    model="gpt-3.5-turbo",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
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


def summarize_paper(paper: arxiv.Result) -> str:
    return gpt_text(
        prompt=f"{paper.summary}",
        system=" ".join(
            [
                IAM,
                "Summarize this abstract in 1 sentence.",
                "Do not explain, only provide the 1 sentence summary.",
                "Here is the abstract for the paper:",
            ]
        ),
    )


def get_embedding(
    paper: arxiv.Result,
    model: str = "text-embedding-ada-002",
) -> np.ndarray:
    embedding: List[float] = openai.Embedding.create(
        model=model,
        input=paper.summary,
    )
    return embedding["data"][0]["embedding"]


class LocalDB(object):

    def __init__(
        self,
        filepath: str = None,
    ):
        self.df = None # one dataframe to rule them all
        if filepath is None:
            log.info("No filepath provided for local DB.")
            filepath = os.path.join(DATA_DIR, "papers.csv")
        self.filepath = filepath
        if os.path.exists(filepath):
            log.info(f"Loading existing local DB from {self.filepath}")
            self.df = pl.read_csv(self.filepath)

    def save(self):
        log.info(f"Saving local DB to {self.filepath}")
        self.df.write_csv(self.filepath)

    def add_paper(
        self,
        paper: arxiv.Result,
        user: str = None,
    ):
        _df = pl.DataFrame(
            {
                "id": paper.get_short_id(),
                "title": paper.title,
                "url": paper.pdf_url,
                "authors": ",".join([author.name for author in paper.authors]),
                "published": paper.published.strftime(DATEFORMAT),
                "abstract": paper.summary,
                "summary": summarize_paper(paper),
                "tags": ",".join(paper.categories),
                "user": user or "",
                "user_submitted_date": datetime.now().strftime(DATEFORMAT),
                "votes": 0,
                "embedding": get_embedding(paper),
            }
        )
        self.df = self.df.vstack(_df)
        self.save()
        return _df

    def get_papers(self, id: str):
        if len(self.df) == 0:
            return None
        try:
            match = self.df.row(by_predicate=(pl.col("id") == id))
        except NoRowsReturnedError:
            return None
        return {column: value for column, value in zip(self.df.columns, match)}

    # TODO: Return with some kind of sorting
    def list_papers(self) -> Dict[str, Any]:
        yield from self.df.iter_rows(named=True)

    def similarity_search(self, paper: arxiv.Result, k: int = 1):
        # TODO: Refactor to account for multiple columns
        embedding_str = get_embedding(paper)
        embedding: np.ndarray = np.array([float(x) for x in embedding_str.split(",")])
        df_embeddings: np.ndarray = np.array([np.array([float(x) for x in e.split(",")]) for e in self.df["embedding"]])
        cosine_sim: np.ndarray = np.dot(embedding, df_embeddings.T)
        top_k_idx: np.ndarray = np.argpartition(cosine_sim, -k)[-k:]
        yield from self.df[top_k_idx].iter_rows(named=True)


async def add_paper(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
    user: str = None,
    **kwargs,
) -> None:
    for paper in find_papers(msg.content):
        log.info(f"Found paper: {paper.title}")
        id = paper.get_short_id()
        if _paper := db.get_papers(id):
            _msg = f"The paper {id} was already posted"
            log.info(_msg)
            await channel.send(_msg)
        else:
            db.add_paper(paper, user=user)
            _msg = f"Adding paper {id}"
            log.info(_msg)
            await channel.send(_msg)
            _msg = "Looking for similar papers..."
            log.info(_msg)
            await channel.send(_msg)
            embeds = []
            for paper_dict in db.similarity_search(paper):
                embeds.append(
                    discord.Embed(
                        title=paper_dict["title"],
                        url=paper_dict["url"],
                        description=paper_dict["summary"],
                    )
                )
            _msg: str = f"Similar papers ({len(embeds)} total)"
            log.info(_msg)
            await channel.send(embeds=embeds)


async def list_papers(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
    **kwargs,
) -> None:
    embeds = []
    for paper_dict in db.list_papers():
        embeds.append(
            discord.Embed(
                title=paper_dict["title"],
                url=paper_dict["url"],
                description=paper_dict["summary"],
            )
        )
    _msg: str = f"Listing papers ({len(embeds)} total)"
    log.info(_msg)
    await channel.send(embeds=embeds)


async def share_sources(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
    **kwargs,
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


async def author_info(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
    llm: str = DEFAULT_LLM,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    **kwargs,
) -> None:
    formatted_name = gpt_text(
        prompt=f"{msg.content}",
        system=" ".join(
            [
                IAM,
                "You extract and format the author name in a message.",
                "Do not explain, only return one name in the format First,Last.",
                "Example message: 'I am interested in the work of John Smith.'",
                "Example response: 'John,Smith'",
            ]
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    embeds = []
    max_papers_per_author = 3
    for i, result in enumerate(arxiv.Search(query=f"au:{formatted_name}").results()):
        if i >= max_papers_per_author:
            break
        embeds.append(
            discord.Embed(
                title=result.title,
                url=result.pdf_url,
                description=summarize_paper(result),
            )
        )
    await channel.send(
        content=f"Here are the other papers by author: {formatted_name}:",
        embeds=embeds[:10],
    )


async def chat(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: LocalDB,
    llm: str = DEFAULT_LLM,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    **kwargs,
) -> None:
    _llm_func: Callable = None
    if llm.startswith("gpt"):
        _llm_func = gpt_text
    elif llm.startswith("palm"):
        _llm_func = palm_text
    response = _llm_func(
        prompt=f"{msg.content}",
        system=IAM,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    await channel.send(response)


@dataclass
class Behavior:
    name: str
    func: Callable
    description: str


class PaperBot(discord.Client):
    # TODO: Every X seconds automatically post to the channel

    # Channel IDs
    CHANNELS: Dict[str, int] = {
        "papers": 1107745177264726036,
        "bot-debug": 1110662456323342417,
    }

    def __init__(
        self,
        *args,
        channel_name: str = "bot-debug",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        default_llm: str = DEFAULT_LLM,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.db = LocalDB()
        if self.CHANNELS.get(channel_name, None) is None:
            raise ValueError(f"Channel {channel_name} not found.")
        self.channel_id: int = self.CHANNELS[channel_name]
        self.max_tokens: int = max_tokens
        self.default_llm: str = default_llm
        self.temperature: float = temperature
        self.behaviors: Dict[str, Behavior] = {}
        # Populate list of action
        for action in [
            Behavior("chat", chat, "Chat with the bot."),
            Behavior("add_paper", add_paper, "Add a paper to the db."),
            Behavior("list_papers", list_papers, "List all papers in the db."),
            Behavior("author_info", author_info, "Shares previous work for an author."),
            Behavior("share_sources", share_sources, "Share links for finding papers."),
        ]:
            self.behaviors[action.name] = action

    async def on_ready(self):
        _msg = f"{EMOJI}{NAME} has entered the chat!"
        log.info(_msg)
        await self.get_channel(self.channel_id).send(_msg)

    async def on_message(self, msg: discord.Message):
        if msg.author.id == self.user.id:
            return
        log.info(f"Received message: {msg.content}")
        if self.user.id in [m.id for m in msg.mentions]:
            log.info(f"Mentioned in message by {msg.author.name}")
            _system_prompt: List[str] = [
                "Choose a behavior from the list below.",
                "Do not explain, return the name of the behavior only.",
            ]
            for action in self.behaviors.values():
                _system_prompt.append(f"{action.name}: {action.description}")
            behavior_guess: str = gpt_text(
                prompt=f"{msg.content}",
                system="".join(_system_prompt),
            )
            if behavior := self.behaviors.get(behavior_guess, None):
                _msg = f"Running behavior: {behavior_guess}."
                log.info(_msg)
                await self.get_channel(self.channel_id).send(_msg)
                await behavior.func(
                    msg,
                    self.get_channel(self.channel_id),
                    self.db,
                    llm=self.default_llm,
                    user=msg.author.name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                _msg = f"Could not find behavior: {behavior_guess}."
                log.warning(_msg)
                await self.get_channel(self.channel_id).send(_msg)

    async def on_error(self, event, *args, **kwargs):
        _msg = f"Error in event {event}."
        log.error(_msg)
        await self.get_channel(self.channel_id).send(_msg)
    
    async def on_disconnect(self):
        _msg = f"{EMOJI}{NAME} has left the chat!"
        log.info(_msg)
        await self.get_channel(self.channel_id).send(_msg)

    

if __name__ == "__main__":
    set_discord_key()
    set_huggingface_key()
    set_openai_key()
    set_palm_key()

    intents = discord.Intents.default()
    intents.message_content = True
    bot = PaperBot(intents=intents)
    bot.run(os.environ["DISCORD_BOT_TOKEN"])
