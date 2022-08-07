import transformers as tr
import logging
from rich.logging import RichHandler
import uvicorn
from fastapi import FastAPI, APIRouter
import argparse as ap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)


class PipelineAPI:

    def __init__(self, model_name: str) -> None:
        self.pipeline = tr.pipeline('token-classification', model=model_name, device="cpu", aggregation_strategy="simples")
        self.router = APIRouter()
        self.router.add_api_route("/predict", self.predict, methods=["POST"])
        self.router.add_api_route("/health", self.health, methods=["GET"])

    def predict(self, texts: list[str]) -> str:
        return self.pipeline(texts)

    def health(self) -> str:
        return "OK"

if __name__ == "__main__":

    parser = ap.ArgumentParser(prog="pipeline")
    parser.add_argument("--model", help="model name", default="Rexhaif/rubert-base-srl-seqlabeling")
    parser.add_argument("--port", help="port", default=8000)
    args: ap.Namespace = parser.parse_args()

    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    del uvicorn_log_config["loggers"][""]

    app = FastAPI()
    api = PipelineAPI(args.model)
    app.include_router(api.router)

    uvicorn.run(app, log_config=uvicorn_log_config, port=args.port, host="0.0.0.0")
