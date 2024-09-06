import subprocess
import urllib
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pyarrow.parquet as pq  # type: ignore
import pandas as pd  # type: ignore

from torch.utils.data import DataLoader  # TODO: upgrade to DataLoader2
from torchdata.datapipes.iter import IterDataPipe  # type: ignore

from retrying import retry

# from feedml.job.entrypoint.config import BatchInferJobConfig
# from feedml.utils.datafile import _find_filesystem

logger = logging.getLogger(__name__)


def _fetch_image_retryable_exceptions(exception: Exception) -> bool:
    # TODO: refine the retryable exceptions
    if "failure in name resolution" in str(exception).lower():
        logger.warning("failure in name resolution, retrying")
        return True
    return False


@retry(
    stop_max_attempt_number=20,
    retry_on_exception=_fetch_image_retryable_exceptions,
    wait_exponential_max=600_000,
)
def fetch_image(image_id: str) -> np.ndarray:
    url = f"https://m.media-amazon.com/images/I/{image_id}.jpg"
    with urllib.request.urlopen(url) as request:
        img_array = np.asarray(bytearray(request.read()), dtype=np.int8)
    return img_array


def process_image(img: np.ndarray, dim: int) -> np.ndarray:
    import cv2  # type: ignore

    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        return img
    img = cv2.resize(img, (int(dim), int(dim)), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class LocalParquetFileLineReaderAmznImageEmbedDataPipe(IterDataPipe):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        pa_dataset = pq.ParquetDataset([self.file_path], filesystem=_find_filesystem(self.file_path))
        df = pa_dataset.read(columns=["asin", "image_id", "marketplace_id"]).to_pandas()
        asins = df["asin"].to_numpy()
        image_ids = df["image_id"].to_numpy()
        for index in range(0, df.shape[0]):
            asin = asins[index]
            image_id = image_ids[index]
            try:
                img = fetch_image(image_id)
                img = process_image(img, 224)
                if img is None:
                    continue
            except Exception as e:
                logger.info(f"Error when processing: {asin}", e)
                continue
            yield {"asin": asin, "image": img}


def model_fn(config: Dict[str, Any]):
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input  # type: ignore

    model_s3_dir = config["from_pretrained"]
    model_dir = "/tmp/feedml/pretrained/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    aws_sync_command = f"aws s3 sync {model_s3_dir} {model_dir}"
    subprocess.run(aws_sync_command, shell=True, check=True)

    if tf.test.gpu_device_name():
        logger.info("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        logger.info("No GPU available. Running on CPU.")
    model = tf.keras.models.load_model(model_dir)
    return model


def batchinfer_fn(
    model,
    input_file: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """

    :param model:
    :param input_file:
    :param config:
    :return:
    """

    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input  # type: ignore

    start_time = time.time()
    dataset = LocalParquetFileLineReaderAmznImageEmbedDataPipe(input_file)
    embeddings_list = []
    asins_list = []
    cnt = 0
    batch_size = 128
    # TODO: move DataLoader as a generic implementation in torchdata_batchinfer
    for batch in DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # disable multiprocess since it's not a real datapipe implementation
    ):
        cnt += 1
        asins = np.array(batch["asin"]).astype(str)
        asins_list.append(asins)
        images = batch["image"].numpy()
        preprocessed_images = preprocess_input(images)
        embeddings = model.predict(preprocessed_images)
        embeddings_list.append(embeddings)
        if cnt % 50 == 0:
            logger.info(f"Elapsed time: {time.time() - start_time} seconds for {cnt * batch_size} records")
    merged_asins = np.concatenate(asins_list, axis=0)
    merged_embeddings = np.concatenate(embeddings_list, axis=0)
    df_asins = pd.DataFrame(merged_asins, columns=["asin"])
    embed_cols = [f"embed_{x}" for x in range(0, 1024)]
    df_flat_embeds = pd.DataFrame(merged_embeddings, columns=embed_cols)
    df = pd.concat([df_asins, df_flat_embeds], axis=1)
    df["embed"] = df[embed_cols].apply(lambda x: np.array(x), axis=1)
    df.drop(columns=embed_cols, inplace=True)
    return df
