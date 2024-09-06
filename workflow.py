import feedml
from feedml.api import FEEDETLDatasource, batch_infer
from feedml.utils.aws_resource import s3_client
from feedml.artifact.api import Artifact
from feedml.utils.dates import yesterday, datetime_obj
import subprocess
import pandas as pd

image_uri = "076944657598.dkr.ecr.us-east-1.amazonaws.com/feedml-tensorflow-andrzhan:latest"
config = {
    "from_pretrained": "s3://sparp-dev/shilinz/image_embedding/models/ResNet50-allGLs-TripletLoss-CosineDist-DenseReLU1024-epoch5-v2/"
}
output_s3_bucket = "sparp-dev"


def asin_image_embed_batch_infer(marketplace, run_date):
    # needed in case upstream data is unavailable, if upstream data becomes available eventually this will be overwritten
    transfer_previous_day_data_to_target_day(marketplace, yesterday(run_date))
    feed_etl_source = FEEDETLDatasource(
        database_name="sparp_feed_prod",
        table_name="daily_ach_tommy_asin_to_image_id",
        marketplace=marketplace,
        start_date=yesterday(run_date),
        end_date=yesterday(run_date),
    )
    feed_etl_source.is_available(max_wait_in_seconds=172800)  # wait for 48 hours
    files = feed_etl_source.list_files()
    input_data_uri = files[0][0 : files[0].rindex("/") + 1]
    response = batch_infer(
        image_uri=image_uri,
        input_data_uri=input_data_uri,
        marketplace=marketplace,
        instance_type="ml.g5.2xlarge",
        instance_count=50,
        region="us-east-1",
        processor="SMCollectiveCommProcessor",
        wait=True,
        config=config,
    )
    run_datetime_yesterday = datetime_obj(yesterday(run_date))
    transfer_to_s3_cache(marketplace, response.artifact_id, run_datetime_yesterday)
    return {}


def transfer_previous_day_data_to_target_day(marketplace, run_date):
    previous_datetime = datetime_obj(yesterday(run_date))
    run_datetime = datetime_obj(run_date)
    prev_s3_prefix = get_output_s3_prefix(marketplace, previous_datetime)
    curr_s3_prefix = get_output_s3_prefix(marketplace, run_datetime)
    list_objs_response = s3_client().list_objects_v2(Bucket=output_s3_bucket, Prefix=curr_s3_prefix)
    if list_objs_response["KeyCount"] == 0:
        input_s3_path = f"s3://{output_s3_bucket}/{prev_s3_prefix}"
        output_s3_path = f"s3://{output_s3_bucket}/{curr_s3_prefix}"
        aws_sync_command = f"aws s3 sync {input_s3_path} {output_s3_path}"
        subprocess.run(aws_sync_command, capture_output=True, shell=True, check=True)


def transfer_to_s3_cache(marketplace, artifact_id, run_datetime):
    input_s3_path = Artifact(artifact_id)._s3_path("")
    output_s3_prefix = get_output_s3_prefix(marketplace, run_datetime)
    output_s3_path = f"s3://{output_s3_bucket}/{output_s3_prefix}"
    # remove any existing embeddings that may have come from Day T-2
    aws_rm_command = f"aws s3 rm {output_s3_path} --recursive"
    subprocess.run(aws_rm_command, capture_output=True, shell=True, check=True)
    # upload the embeddings from Day T-1
    aws_sync_command = f"aws s3 sync {input_s3_path} {output_s3_path}"
    subprocess.run(aws_sync_command, capture_output=True, shell=True, check=True)
    if not has_parquet(output_s3_bucket, output_s3_prefix):
        add_empty_parquet_to_cache(output_s3_bucket, output_s3_prefix)


def get_output_s3_prefix(marketplace, run_datetime):
    return f"image_embedding/{marketplace}/year={run_datetime.year}/month={run_datetime.month:02d}/day={run_datetime.day:02d}/"


def has_parquet(s3_bucket, s3_prefix):
    list_objs_response = s3_client().list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    files = list_objs_response["Contents"]
    for file in files:
        key = file["Key"]
        if ".parquet" in key:
            return True
    return False


def add_empty_parquet_to_cache(s3_bucket, s3_prefix):
    empty_df = pd.DataFrame()
    empty_df_filename = "empty_df.parquet"
    empty_df.to_parquet(empty_df_filename)
    s3_client().upload_file(Filename=empty_df_filename, Bucket=s3_bucket, Key=f"{s3_prefix}{empty_df_filename}")


@feedml.workflow(schedule_cadence="daily")
class MyWorkflow:
    @feedml.step()
    def asin_image_embed_batch_infer_uk(self, params, depends_on):
        run_date = params["FEEDML_SCHEDULED_TIME"]
        return asin_image_embed_batch_infer("UK", run_date)

    @feedml.step()
    def asin_image_embed_batch_infer_de(self, params, depends_on):
        run_date = params["FEEDML_SCHEDULED_TIME"]
        return asin_image_embed_batch_infer("DE", run_date)
