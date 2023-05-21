import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Dict, Iterator, List, Union

import arxiv
import discord
import google.generativeai as genai
import openai
import transformers

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("paperbot")
formatter = logging.Formatter("🗃️|%(asctime)s|%(message)s")
# Set up console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)
# Set up file handler
fh = logging.FileHandler(f'_paperbot_{datetime.now().strftime("%D%M%Y")}.log')
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
                key = f.read()
        except FileNotFoundError:
            log.warning("Discord API key not found.")
            return
    os.environ["DISCORD_API_KEY"] = key
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

    genai.configure(api_key=key)
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


def palm_chat(prompt, context, examples=None):
    """https://developers.generativeai.google/tutorials/chat_quickstart"""
    # # An array of "ideal" interactions between the user and the model
    # examples = [
    #     ("What's up?", # A hypothetical user input
    #     "What isn't up?? The sun rose another day, the world is bright, anything is possible! ☀️" # A hypothetical model response
    #     ),
    #     ("I'm kind of bored",z
    #     "How can you be bored when there are so many fun, exciting, beautiful experiences to be had in the world? 🌈")
    # ]
    response = palm.chat(
        context=context,
        examples=examples,
        messages=prompt,
    )
    return response.last


def find_papers(msg: str) -> Iterator[arxiv.Result]:
    pattern = r"arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)"
    matches = re.findall(pattern, msg)
    for match in matches:
        arxiv_id = match.group(1)
        search = arxiv.Search(id_list=[arxiv_id])
        for paper in search.results():
            yield paper


def paper_blurb(paper: arxiv.Result) -> str:
    title = paper.title
    authors: List[str] = [author.name for author in paper.authors]
    published = paper.published.strftime("%m/%d/%Y")
    url = paper.pdf_url
    blurb = f"""
----- 📝 ArXiV -----
[{title}]({url})
{published}
{", ".join(authors)}
--------------------
"""
    return blurb


def gpt_chat(context, prompt, examples=None):
    # TODO: examples converts tuples into gpt dict format
    return gpt_text(prompt, system=context)


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


class FakeDB(object):
    def __init__(self):
        self.papers = {}

    def add_paper(self, paper: arxiv.Result):
        self.papers[paper.get_short_id()] = paper

    def get_papers(self, id: str):
        return self.papers[id]


class PaperBot(discord.Client):
    """
    https://github.com/Rapptz/discord.py/tree/master/examples


    on message
    - check for arxiv link
    - if link, get paper blurb
    - send paper blurb
    - start post task

    post task:
    - add paper to priority queue, priority based on votes?
    - add paper to static db (txt file? notion? google big table? google sheets?)
    - post current paper queue to channel (every X amount of time?)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = FakeDB()

    async def on_ready(self):
        log.info(f"We have logged in as {self.user}")

    async def on_message(self, msg):
        # we do not want the bot to reply to itself
        if msg.author.id == self.user.id:
            return
        log.debug(f"Received message: {msg.content}")
        for paper in find_papers(msg.content):
            log.info(f"Found paper: {paper.title}")
            id = paper.get_short_id()
            if self.db.get_papers(id) is None:
                self.db.add_paper(paper)
            else:
                log.info(f"Paper {id} already in DB, skipping.")
                continue
            blurb = paper_blurb(paper)
            await msg.channel.send(blurb)


if __name__ == "__main__":
    set_discord_key()
    set_huggingface_key()
    set_openai_key()
    set_palm_key()

    intents = discord.Intents.default()
    intents.message_content = True
    client = PaperBot(intents=intents)
    client.run(os.environ["DISCORD_API_KEY"])
