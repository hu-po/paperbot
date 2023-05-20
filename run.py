import logging
import os
import uuid
from io import BytesIO
from datetime import datetime
from typing import Any, Dict, List, Union
import re
import arxiv
import discord
from discord.ext import commands

import requests

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
    import openai
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
    import google.generativeai as genai
    genai.configure(api_key=key)
    log.info("Palm API key set.")

def palm_text(prompt):
    """https://developers.generativeai.google/tutorials/text_quickstart"""
    import google.generativeai as palm
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
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
    import google.generativeai as palm

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

def find_paper(url: str) -> arxiv.Result:
    pattern = r"arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)"
    match = re.search(pattern, url)
    if match:
        arxiv_id = match.group(1)
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        return paper
    else:
        return None

def paper_blurb(paper: arxiv.Result) -> str:
    title = paper.title
    authors: List[str] = [author.name for author in paper.authors]
    published = paper.published.strftime("%m/%d/%Y")
    url = paper.pdf_url
    blurb = f"""
----- 📝 ArXiV -----
{url}
{title}
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
    import openai
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


class PaperBot(discord.Client):
    async def on_ready(self):
        log.info(f'We have logged in as {self.user}')

    async def on_msg(self, msg):
        if msg.author == self.user:
            return

        arxiv_link_pattern = r'(https?:\/\/arxiv\.org\/[a-z]+\/[\w\.]+)'
        links = re.findall(arxiv_link_pattern, msg.content)

        if links:
            await msg.channel.send(f'I found the following arXiv links in your msg: {", ".join(links)}')



if __name__ == "__main__":
    
    set_discord_key()
    # set_huggingface_key()
    # set_openai_key()
    # set_palm_key()

    intents = discord.Intents.default()
    intents.message_content = True
    client = PaperBot(intents=intents)
    client.run(os.environ["DISCORD_API_KEY"])
