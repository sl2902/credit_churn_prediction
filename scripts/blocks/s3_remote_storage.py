import os

from prefect.filesystems import RemoteFileSystem

# from dotenv import load_dotenv


# load_dotenv(".env")

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "credit-churn-experiment")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL", "http://127.0.0.1:9000")

minio_block = RemoteFileSystem(
    basepath=f"s3://{EXPERIMENT_NAME}/prefect",
    settings={
        "key": AWS_ACCESS_KEY_ID,
        "secret": AWS_SECRET_ACCESS_KEY,
        "client_kwargs": {"endpoint_url": AWS_S3_ENDPOINT_URL},
    },
)
minio_block.save("minio", overwrite=True)
