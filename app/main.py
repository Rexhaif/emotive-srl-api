import json
import logging

from fastapi import FastAPI

from .config import settings
from .pipeline import PipelineAPI
from .utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

with open("./resources/good_lemmas.json", "r") as f:
    good_lemmas = json.load(f)

app = FastAPI()
api = PipelineAPI(settings.model.name, good_lemmas=good_lemmas)
app.include_router(api.router)
