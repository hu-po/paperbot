import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial, wraps
from typing import Any, Callable, Dict, Iterator, List, Union

import arxiv
import discord
import google.generativeai as palm
import numpy as np
import openai
import polars as pl
from polars.exceptions import NoRowsReturnedError

NAME: str = "paperbot"
EMOJI: str = "🗃️"
IAM: str = "".join(
    [
        f"You are {NAME}, a arxiv chatbot.",
        "You use lots of emojis.",
    ]
)
DATEFORMAT = "%d.%m.%y"
SOURCES: Dict[str, str] = {
    # TODO: Scrape latest info?
    "Twitter": "https://twitter.com/i/lists/1653485531546767361",
    "PapersWithCode": "https://paperswithcode.com/",
    "Reddit": "https://www.reddit.com/user/deephugs/m/ml/",
    "ArxivSanity": "http://www.arxiv-sanity.com/",
    "LabML": "https://papers.labml.ai/papers/weekly/",
}

DEFAULT_LLM: str = "gpt-3.5-turbo"
assert DEFAULT_LLM in [
    "gpt-3.5-turbo",
    "gpt-4",
    "palm",
    # TODO: llama through hf?
]
DEFAULT_TEMPERATURE: float = 0
DEFAULT_MAX_TOKENS: int = 64

# DEBUG_MODE: bool = True
DEBUG_MODE: bool = False

# Normal Configuration
DEBUG_LEVEL: int = logging.INFO
DISCORD_CHANNEL: int = 1107745177264726036  # papers
LIFESPAN: timedelta = timedelta(days=3)
GREETING_MESSAGE_ENABLED: bool = True
AUTO_MESSAGE_ENABLED: bool = False
AUTO_MESSAGES_INTERVAL: timedelta = timedelta(hours=1)
HEARTBEAT_INTERVAL: timedelta = timedelta(seconds=60)
MAX_MESSAGES: int = 4
MAX_MESSAGES_INTERVAL: timedelta = timedelta(seconds=10)
# Debug Configuration
if DEBUG_MODE:
    DEBUG_LEVEL: int = logging.DEBUG
    DISCORD_CHANNEL: int = 1110662456323342417  # bot-debug
    LIFESPAN: timedelta = timedelta(minutes=5)
    HEARTBEAT_INTERVAL: timedelta = timedelta(seconds=3)
    MAX_MESSAGES: int = 5
    MAX_MESSAGES_INTERVAL: timedelta = timedelta(minutes=1)
    AUTO_MESSAGES_INTERVAL: timedelta = timedelta(minutes=1)
    GREETING_MESSAGE_ENABLED: bool = True
    AUTO_MESSAGE_ENABLED: bool = True

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_DIR = os.path.join(ROOT_DIR, ".keys")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILENAME: str = f"{NAME}{datetime.now().strftime(DATEFORMAT)}.{EMOJI}"
DB_FILEPATH: str = os.path.join(DATA_DIR, DB_FILENAME)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=DEBUG_LEVEL)
log = logging.getLogger(NAME)
formatter = logging.Formatter("%(asctime)s|%(message)s")
# Set up console handler
ch = logging.StreamHandler()
ch.setLevel(DEBUG_LEVEL)
ch.setFormatter(formatter)
log.addHandler(ch)
# Set up file handler
logfile_name = f"_paperbot_{datetime.now().strftime(DATEFORMAT)}.log"
logfile_path = os.path.join(LOG_DIR, logfile_name)
fh = logging.FileHandler(logfile_path)
fh.setLevel(DEBUG_LEVEL)
fh.setFormatter(formatter)
log.addHandler(fh)

log.debug(f"ROOT_DIR: {ROOT_DIR}")
log.debug(f"KEYS_DIR: {KEYS_DIR}")
log.debug(f"DATA_DIR: {DATA_DIR}")
log.debug(f"LOG_DIR: {LOG_DIR}")


def time_and_log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        log.debug(f"Calling: {func.__name__} - duration: {duration:.2f}")
        return result

    return wrapper


@time_and_log
def set_openai_key(key=None) -> str:
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "openai.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("OpenAI API key not found.")
            return
    openai.api_key = key
    log.debug("OpenAI API key set.")
    return key


@time_and_log
def set_discord_key(key=None) -> str:
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "discord.txt"), "r") as f:
                return f.read()
        except FileNotFoundError:
            log.warning("Discord API key not found.")
            return
    log.debug("Discord API key set.")


@time_and_log
def set_huggingface_key(key=None) -> str:
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "huggingface.txt"), "r") as f:
                return f.read()
        except FileNotFoundError:
            log.warning("HuggingFace API key not found.")
            return
    log.debug("HuggingFace API key set.")


@time_and_log
def set_palm_key(key=None) -> str:
    if key is None:
        try:
            with open(os.path.join(KEYS_DIR, "palm.txt"), "r") as f:
                key = f.read()
        except FileNotFoundError:
            log.warning("Palm API key not found.")
            return
    palm.configure(api_key=key)
    log.debug("Palm API key set.")
    return key


def default_behavior_parameters():
    return {
        "type": "object",
        "properties": {},
        "required": [],
    }


@dataclass
class Behavior:
    name: str
    func: Callable
    description: str
    # Return type of OpenAI Functions API
    parameters: Dict[str, Any] = field(default_factory=default_behavior_parameters)
    # {
    #     "type": "object",
    #     "properties": {
    #         "location": {
    #             "type": "string",
    #             "description": "The city and state, e.g. San Francisco, CA",
    #         },
    #         "unit": {
    #             "type": "string",
    #             "enum": ["celsius", "fahrenheit"],
    #         },
    #     },
    #     "required": ["location"],
    # }


@time_and_log
def find_paperid_in_msg(msg: str) -> Iterator[arxiv.Result]:
    pattern = r"arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)"
    matches = re.findall(pattern, msg)
    for arxiv_id in matches:
        for paper in arxiv.Search(id_list=[arxiv_id]).results():
            yield paper


@time_and_log
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


@time_and_log
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


@time_and_log
def gpt_function(
    prompt: str = None,
    model="gpt-3.5-turbo-0613",
    functions: List[str] = None,
    behaviors: Dict[str, Behavior] = None,
) -> Union[None, Callable]:
    log.debug(f"Function call to GPT {model}: \n {prompt}")
    response: Dict = openai.ChatCompletion.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        functions=functions,
        function_call="auto",
    )
    message: Dict = response["choices"][0]["message"]
    # Step 2, check if the model wants to call a function
    if message.get("function_call", None):
        _func_info: Dict = message["function_call"]
        behavior_name: str = _func_info.get("name", None)
        log.debug(f"Function call detected {behavior_name}")
        if behavior := behaviors.get(behavior_name, None):
            log.info(f"Calling behavior {behavior_name}")
            return partial(behavior.func, **json.loads(_func_info["arguments"]))
        else:
            log.warning(f"Function {behavior_name} not found in function_dict.")
            return None
    log.warning("No function call detected in GPT response.")
    return None


@time_and_log
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


@time_and_log
def get_embedding(
    paper: arxiv.Result,
    model: str = "text-embedding-ada-002",
) -> np.ndarray:
    embedding: List[float] = openai.Embedding.create(
        model=model,
        input=paper.summary,
    )
    return embedding["data"][0]["embedding"]


class TinyDB:
    """A tiny database that is secretly a Polars dataframe in a CSV file."""

    def __init__(
        self,
        filepath: str = DB_FILEPATH,
    ):
        self.df = None  # one dataframe to rule them all
        self.filepath = filepath
        if os.path.exists(filepath):
            log.info(f"Loading existing local DB from {self.filepath}")
            self.df = pl.read_csv(self.filepath)

    def save(self, df: pl.DataFrame = None):
        if df is not None:
            self.df = df
        self.df.write_csv(self.filepath)
        log.info(f"Saved local DB to {self.filepath}")

    def add_paper(
        self,
        paper: arxiv.Result,
        user: str = None,
    ):
        _data = {
            "id": paper.get_short_id(),
            "title": paper.title,
            "url": paper.pdf_url,
            "authors": ",".join([author.name for author in paper.authors]),
            "published": paper.published.strftime(DATEFORMAT),
            "abstract": paper.summary,
            "summary": summarize_paper(paper),
            "tags": ",".join(paper.categories),
            "user_submitted_date": datetime.now().strftime(DATEFORMAT),
            "votes": str(user) or "",
            "votes_count": 1,
        }
        for i, val in enumerate(get_embedding(paper)):
            _data[f"embedding_{i}"] = val
        _df = pl.DataFrame(_data)
        if self.df is None:
            self.df = _df
        else:
            self.df = self.df.vstack(_df)
        self.save()
        return _df

    def get_papers(self, id: str):
        if self.df is None or len(self.df) == 0:
            return None
        try:
            match = self.df.row(by_predicate=(pl.col("id") == id))
        except NoRowsReturnedError:
            return None
        return {column: value for column, value in zip(self.df.columns, match)}

    def similarity_search(self, paper: arxiv.Result, k: int = 3):
        if self.df is None or len(self.df) == 0:
            return None
        k = min(k, len(self.df))
        embedding: List[float] = get_embedding(paper)
        embedding: np.ndarray = np.array(embedding)
        df_embeddings: np.ndarray = np.array(
            self.df[[f"embedding_{x}" for x in range(1536)]]
        )
        cosine_sim: np.ndarray = np.dot(embedding, df_embeddings.T)
        # Create new Polars dataframe with cosine similarity as column
        _df = self.df[["title", "url", "summary"]]
        _df = _df.with_columns(pl.from_numpy(cosine_sim, schema=["cosine_sim"]))
        # Sort by cosine similarity
        _df = _df.sort(by="cosine_sim", descending=True)
        # Return the top k rows, but skip the first row
        yield from _df.head(k + 1).tail(k).iter_rows(named=True)


@time_and_log
async def add_paper(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: TinyDB,
    **kwargs,
) -> None:
    for paper in find_paperid_in_msg(msg.content):
        log.info(f"Found paper: {paper.title}")
        id = paper.get_short_id()
        if _paper := db.get_papers(id):
            _msg = f"The paper {id} was already posted"
            log.info(_msg)
            await channel.send(_msg)
        else:
            db.add_paper(paper, user=str(msg.author.id))
            _msg = f"Adding paper {id}"
            log.info(_msg)
            await channel.send(_msg)
            _msg = "Looking for similar papers..."
            log.info(_msg)
            await channel.send(_msg)
            embeds = []
            for _paper in db.similarity_search(paper):
                if _paper["title"] == paper.title:
                    continue
                embeds.append(
                    discord.Embed(
                        title=_paper["title"],
                        url=_paper["url"],
                        description=f"Similarity: {_paper['cosine_sim']:.2f}\n\n"
                        f"{_paper['summary']}",
                    )
                )
            if len(embeds) == 0:
                continue
            _msg: str = f"Similar papers ({len(embeds)} total)"
            log.info(_msg)
            await channel.send(embeds=embeds)


@time_and_log
async def list_papers(
    message: discord.Message,
    channel: discord.TextChannel,
    db: TinyDB,
    **kwargs,
) -> None:
    embeds = []
    if db.df is None or len(db.df) == 0:
        _msg = "No papers in database."
        await channel.send(_msg)
        log.info(f"Sending message: {_msg}")
        return
    num_papers = kwargs.get("num_papers", 4)
    _df: pl.DataFrame = db.df
    if sort_by := kwargs.get("sort_by"):
        if sort_by in ["votes", "votes_count", "voting"]:
            sort_by = "votes_count"
        if sort_by in db.df.columns:
            _df = _df.sort(by=sort_by, descending=True)
        else:
            _msg = f"Trying to sort by column '{sort_by}' which doesn't exist."
            await channel.send(_msg)
            log.info(f"Sending message: {_msg}")
            return
    for i, paper_dict in enumerate(_df.iter_rows(named=True)):
        if i >= num_papers:
            break
        embeds.append(
            discord.Embed(
                title=paper_dict["title"],
                url=paper_dict["url"],
                description=f"votes: {paper_dict['votes_count']}\n\n"
                f"{paper_dict['summary']}",
            )
        )
    _msg: str = f"Found papers, listing ({len(embeds)} total)"
    await channel.send(embeds=embeds)
    log.info(f"Sending message: {_msg}")


@time_and_log
async def vote_for_paper(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: TinyDB,
    **kwargs,
) -> None:
    for paper in find_paperid_in_msg(msg.content):
        log.info(f"Found paper: {paper.title}")
        paper_id = paper.get_short_id()
        paper_mask = db.df["id"] == paper_id
        if paper_mask.sum() == 0:
            _msg = f"Paper {paper_id} is not in the database."
            await channel.send(_msg)
            log.info(f"Sending message: {_msg}")
            return
        _df: pl.DataFrame = db.df.filter(paper_mask).head(1)
        votes_raw: str = str(_df["votes"].to_list()[0])
        votes: List[str] = []
        if len(votes_raw) > 0:
            votes = [str(_) for _ in votes_raw.split(",")]
        user_id = str(msg.author.id)
        if user_id in votes:
            _msg = f"{msg.author.display_name} has already voted for this paper."
            await channel.send(_msg)
            log.info(f"Sending message: {_msg}")
            return
        votes.append(user_id)
        _df = _df.with_columns(pl.col("votes").apply(lambda _: ",".join(votes)))
        _df = _df.with_columns(pl.col("votes_count").apply(lambda _: len(votes)))
        db.save(db.df.update(_df))
        _msg = f"User {msg.author.display_name} has voted for paper {paper_id}."
        await channel.send(_msg)
        log.info(f"Sending message: {_msg}")


@time_and_log
async def share_sources(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: TinyDB,
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
    log.debug("Listing the sources.")
    await channel.send(content="Here are some sources for papers:", embeds=embeds)


@time_and_log
async def author_info(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: TinyDB,
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


@time_and_log
async def chat(
    msg: discord.Message,
    channel: discord.TextChannel,
    db: TinyDB,
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


class PaperBot(discord.Client):
    def __init__(
        self,
        *args,
        channel_id: str = DISCORD_CHANNEL,
        lifespan: timedelta = LIFESPAN,
        heartbeat_interval: timedelta = HEARTBEAT_INTERVAL,
        max_messages: int = MAX_MESSAGES,
        max_messages_interval: timedelta = MAX_MESSAGES_INTERVAL,
        auto_message_interval: timedelta = AUTO_MESSAGES_INTERVAL,
        auto_message_enabled: bool = AUTO_MESSAGE_ENABLED,
        greeting_message_enabled: bool = GREETING_MESSAGE_ENABLED,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        default_llm: str = DEFAULT_LLM,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.db = TinyDB()
        self.channel_id: int = channel_id
        self.max_tokens: int = max_tokens
        self.default_llm: str = default_llm
        self.temperature: float = temperature
        self.lifespan: timedelta = lifespan
        self.birthdate: datetime = datetime.now()
        self.heartbeat_interval: timedelta = heartbeat_interval
        self.max_messages: int = max_messages
        self.max_messages_interval: timedelta = max_messages_interval
        self.message_cache: Dict[str, discord.Message] = {}
        self.greetings_message_enabled: bool = greeting_message_enabled
        self.auto_message_enabled: bool = auto_message_enabled
        self.auto_message_interval: timedelta = auto_message_interval
        self.last_auto_message: datetime = datetime.now()
        self.behaviors: Dict[str, Behavior] = {}
        self.functions: List[Dict] = []
        # Populate list of action
        for action in [
            Behavior(
                name="chat",
                func=chat,
                description="Chat with the user message.",
            ),
            Behavior(
                name="add_paper",
                func=add_paper,
                description="Add a paper to the db.",
            ),
            Behavior(
                name="vote_for_paper",
                func=vote_for_paper,
                description="User votes for a paper.",
            ),
            Behavior(
                name="list_papers",
                func=list_papers,
                description="List all papers in the db.",
                parameters={
                    "type": "object",
                    "properties": {
                        "sort_by": {
                            "type": "string",
                            "description": "Sort by field.",
                        },
                        "num_papers": {
                            "type": "integer",
                            "description": "Number of papers to list.",
                            "default": 3,
                        },
                    },
                    "required": [],
                },
            ),
            Behavior(
                name="author_info",
                func=author_info,
                description="Shares previous work for an author.",
            ),
            Behavior(
                name="share_sources",
                func=share_sources,
                description="Share links for finding papers.",
            ),
        ]:
            self.behaviors[action.name] = action
            self.functions.append(
                {
                    "name": action.name,
                    "description": action.description,
                    "parameters": action.parameters,
                }
            )

    async def on_ready(self):
        if self.greetings_message_enabled:
            _msg: str = gpt_text(
                prompt="Say hello to the chat",
                system=IAM,
                temperature=1,
            )
            await self.get_channel(self.channel_id).send(_msg)
            log.info(f"Sending message: {_msg}")

    async def setup_hook(self) -> None:
        self.bg_task = self.loop.create_task(self.heartbeat())

    async def heartbeat(self):
        await self.wait_until_ready()
        while not self.is_closed():
            # TODO: Return priority queue for papers
            # TODO: Weekly greetings, greetings based on dates/seasons
            # TODO: Render the csv file to markdown and post in on a github
            #       repo automatically `from github import Github`
            # TODO: Announcements on the start of every week and the voting results.
            # Programmed death
            if datetime.now() - self.birthdate > self.lifespan:
                _msg: str = gpt_text(
                    prompt=f"You have been a good {IAM}."
                    "Say your goodbyes to your friends.",
                    system=IAM,
                    temperature=1,
                )
                await self.get_channel(self.channel_id).send(_msg)
                log.info(f"Sent goodbye message: {_msg}")
                await self.close()
            # Send message to the channel if it's been a while
            if self.auto_message_enabled and datetime.now() - self.last_auto_message >= self.auto_message_interval:
                _msg: str = gpt_text(
                    prompt="Say something short and funny.",
                    system=IAM,
                    temperature=1,
                )
                await self.get_channel(self.channel_id).send(_msg)
                log.info(f"Sent auto message: {_msg}")
            # Keep a small cache of recent messages
            for _datetime, _msg in self.message_cache.items():
                if datetime.now() - _datetime < self.max_messages_interval:
                    continue
                log.info(f"Removing stale message: {_msg}")
                del self.message_cache[_datetime]
            await asyncio.sleep(self.heartbeat_interval.total_seconds())

    async def on_message(self, msg: discord.Message):
        if msg.author.id == self.user.id:
            log.info("Ignoring message from self.")
            return
        if self.user.id in [m.id for m in msg.mentions]:
            log.debug(f"Mentioned in message by {msg.author.display_name}.")
            if len(self.message_cache) >= self.max_messages:
                _msg = f"I am busy {msg.author.display_name}, please wait."
                await self.get_channel(self.channel_id).send(_msg)
                log.info(_msg)
                return
            self.message_cache[datetime.now()] = msg
            if _callable := gpt_function(
                prompt=f"{msg.content}",
                model="gpt-3.5-turbo-0613",
                functions=self.functions,
                behaviors=self.behaviors,
            ):
                _msg = f"I am going to {_callable.func.__name__}!"
                await self.get_channel(self.channel_id).send(_msg)
                log.debug(f"Calling {_callable.func.__name__}...")
                await _callable(
                    msg,
                    self.get_channel(self.channel_id),
                    self.db,
                    # TODO: accept message cache for chat
                    llm=self.default_llm,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                _msg = "Could not recognize a behavior from your message. Here is what I can do:\n"
                for _behavior in self.behaviors.values():
                    _msg += f"- {_behavior.name}: {_behavior.description}\n"
                await self.get_channel(self.channel_id).send(_msg)

    async def on_disconnect(self):
        await self.get_channel(self.channel_id).send("🪦")
        log.info("Disconnected.")


if __name__ == "__main__":
    log.info(f"Starting at {datetime.now().strftime(DATEFORMAT)}")
    log.info("Setting keys...")
    set_huggingface_key()
    set_openai_key()
    set_palm_key()
    log.info("Starting bot...")
    intents = discord.Intents.default()
    intents.message_content = True
    bot = PaperBot(intents=intents)
    bot.run(set_discord_key())
