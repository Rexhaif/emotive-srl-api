import transformers as tr
import logging
from rich.logging import RichHandler
import numpy as np
from timeit import repeat
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    device = torch.device("cpu")
    pipeline = tr.pipeline("token-classification", model=settings.model.name, device=device, aggregation_strategy='simple')
    res = repeat('_ = pipeline("Мама расстроилась на мальчика")', globals=globals(), number=100, repeat=30)
    res = list(map(lambda x: x / 100, res))
    logger.info(f"Run time: {np.mean(res)} +- {np.std(res)}")
