from typing import List, Dict, Any
import urllib
import os
import logging

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import pandas as pd  # type: ignore

from torchdata.datapipes.iter import IterDataPipe  # type: ignore

from feedml.artifact.api import Artifact
from feedml.job.entrypoint import entrypoint
from feedml.metadata import Metadata
from feedml.processor.distributed import host_world_size, host_rank
from feedml.job.entrypoint.config import BatchInferJobConfig
from feedml.utils.aws_resource import s3_prefix_paginator, s3_client, s3_path_parser
from feedml.utils.datafile import list_dir
from feedml.utils.iterables import partition_list
from feedml.utils.logging import condense_list
from feedml.utils.string import timed_random_str
from feedml.utils.tracker import ContextTracker

logger = logging.getLogger(__name__)


def calculate_inference_candidates(checkpoint_local_dir: str, s3path: str) -> List[str]:
    """
    The inference task is calculated by the following logic:
    1. get checkpoints directory of the artifact
    2. list all files in the input s3 path, and subtract all files in the checkpoints
    3. sort the file names, and use mod host_world_size and host_rank to get the local copy

    :return: A list of S3 paths (i.e. s3://abc/def)
    """
    ckpt_files = list_dir(checkpoint_local_dir, suffix=".ckpt.log")
    processed_files = []
    for fname in ckpt_files:
        with open(fname, "r") as f:
            processed_files.extend(f.readlines())

    processed_files = [e.strip() for e in processed_files]  # remove '\n'
    bucket, prefix = s3_path_parser(s3path)
    all_candidates = [f"s3://{bucket}/{k}" for k in s3_prefix_paginator(bucket, prefix) if k.endswith(".parquet")]

    remaining_candidates = list(set(all_candidates) - set(processed_files))
    remaining_candidates.sort()

    total_hosts, host_index = host_world_size(), host_rank()
    current_host_candidates = partition_list(remaining_candidates, num_partition=total_hosts, length_check=False)[
        host_index
    ]
    logger.info(
        f"The current host ({host_index}/{total_hosts}) will process "
        f"{len(current_host_candidates)}/{len(remaining_candidates)} files: "
        f"{condense_list(current_host_candidates)}"
    )
    return current_host_candidates


def import_model_function():
    """Load model function from users' experiment source.
    It will look for function named `model_fn` inside model.py file."""
    try:
        # try to load the model function at current working directory (i.e. /opt/ml/code)
        import model  # type: ignore
        from model import model_fn  # type: ignore

        # TODO: add assertion on function signature
        logger.info("Loaded model_fn function from current working directory")
        return model_fn
    except ImportError as e:
        logger.exception("Failed to load `model_fn` from user provided experiment source.")
        raise e


def import_inference_function():
    """Load model function from users' experiment source.
    It will look for function named `batchinfer_fn` inside model.py file."""
    try:
        # try to load the model function at current working directory (i.e. /opt/ml/code)
        import model  # type: ignore
        from model import batchinfer_fn  # type: ignore

        # TODO: add assertion on function signature
        logger.info("Loaded batchinfer_fn function from current working directory")
        return batchinfer_fn
    except ImportError as e:
        logger.exception("Failed to load `batchinfer_fn` from user provided experiment source.")
        raise e


@entrypoint(config_type=BatchInferJobConfig)
def main(config: BatchInferJobConfig, artifact: Artifact, metadata: Metadata, tracker: ContextTracker):
    model_fn = import_model_function()
    model = model_fn(config.user_config)
    inference_fn = import_inference_function()

    checkpoint_local_dir = artifact.get_dir("checkpoints")
    inference_candidates = calculate_inference_candidates(checkpoint_local_dir, config.candidate_dataset)
    total_candidates = len(inference_candidates)

    download_dir = artifact.new_dir("download", track=False)
    new_checkpoint_file = os.path.join(checkpoint_local_dir, f"{timed_random_str(rand_len=10)}.ckpt.log")
    for idx, input_s3path in enumerate(inference_candidates):
        logger.info(f"Start processing {idx + 1}/{total_candidates} file: {input_s3path}")
        base_filename = os.path.basename(input_s3path)
        local_filename = os.path.join(download_dir, base_filename)
        bucket, prefix = s3_path_parser(input_s3path)
        s3_client().download_file(bucket, prefix, local_filename)
        output_file = artifact.new_file(base_filename)

        df = inference_fn(model, local_filename, config.user_config)

        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
        logger.info(f"Finished processing {idx + 1}/{total_candidates} file: {input_s3path}")
        with open(new_checkpoint_file, "a+") as f:
            f.write(f"{input_s3path}\n")
        # TODO: make a continuous uploading thread in the background
        artifact.upload()
        logger.info(f"Finished uploading {idx + 1}/{total_candidates} file: {input_s3path}")


if __name__ == "__main__":
    main()
