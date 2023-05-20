import logging
import os
import uuid
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Union
import re
import arxiv
import discord

import requests
from PIL import Image

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
    response = openai.ChatCompletion.create(
        messages=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    return response["choices"][0]["message"]["content"]


def gpt_emoji(prompt):
    try:
        emoji = gpt_text(
            prompt=prompt,
            system=" ".join(
                [
                    "Respond with a single emoji based on the user prompt.",
                    "Respond with only basic original emoji.",
                    "Respond with only one emoji.",
                    "Do not explain, respond with the single emoji only.",
                ]
            ),
            temperature=0.3,
        )
    except Exception:
        emoji = "👻"
    return emoji


def gpt_color():
    try:
        color_name = gpt_text(
            system=" ".join(
                [
                    "You generate unique and interesting colors for a crayon set.",
                    "Crayon color names are only a few words.",
                    "Respond with the colors only: no extra text or explanations.",
                ]
            ),
            temperature=0.99,
        )
        rgb = gpt_text(
            prompt=color_name,
            system=" ".join(
                [
                    "You generate RGB color tuples for digital art based on word descriptions.",
                    "Respond with three integers in the range 0 to 255 representing R, G, and B.",
                    "The three integers should be separated by commas, without spaces.",
                    "Respond with the colors only: no extra text or explanations.",
                ]
            ),
            temperature=0.1,
        )
        rgb = rgb.split(",")
        assert len(rgb) == 3
        rgb = tuple([int(x) for x in rgb])
        assert all([0 <= x <= 256 for x in rgb])
    except Exception:
        color_name = "black"
        rgb = (0, 0, 0)
    return rgb, color_name


def gpt_image(
    prompt: str,
    n: int = 1,
    image_size="512x512",
):
    import openai
    log.debug(f"Image call to GPT with: \n {prompt}")
    response = openai.Image.create(
        prompt=prompt,
        n=n,
        size=image_size,
    )
    img_url = response["data"][0]["url"]
    image = Image.open(BytesIO(requests.get(img_url).content))
    # Output path for original image
    image_name = uuid.uuid4()
    image_path = os.path.join(DATA_DIR, f"{image_name}.png")
    image.save(image_path)
    return image_path

if __name__ == "__main__":
    pass
