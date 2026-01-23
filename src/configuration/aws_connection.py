# import boto3
# import os
# from src.constants import AWS_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID, AWS_REGION_NAME
# from src.logger import logger

# class S3Client:

#     s3_client=None
#     s3_resource = None
#     def __init__(self, region_name=AWS_REGION_NAME):
#         if S3Client.s3_resource==None or S3Client.s3_client==None:
#             __access_key_id = AWS_ACCESS_KEY_ID
#             __secret_access_key = AWS_SECRET_ACCESS_KEY
#             if __access_key_id is None:
#                 raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID} is not not set.")
#             if __secret_access_key is None:
#                 raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY} is not set.")
#             logger.debug("Connection made successfully.")
        
#             S3Client.s3_resource = boto3.resource('s3',
#                                             aws_access_key_id=__access_key_id,
#                                             aws_secret_access_key=__secret_access_key,
#                                             region_name=region_name
#                                             )
#             S3Client.s3_client = boto3.client('s3',
#                                         aws_access_key_id=__access_key_id,
#                                         aws_secret_access_key=__secret_access_key,
#                                         region_name=region_name
#                                         )
#         self.s3_resource = S3Client.s3_resource
#         self.s3_client = S3Client.s3_client

import boto3
from src.constants import AWS_REGION_NAME
from src.logger import logger


class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, region_name=AWS_REGION_NAME):
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            logger.debug("Initializing S3 client using IAM Role / default credential chain")

            # ✅ DO NOT pass access keys
            # boto3 will automatically use:
            # - IAM Role (IMDSv2)
            # - or env vars if present
            S3Client.s3_resource = boto3.resource(
                "s3",
                region_name=region_name
            )

            S3Client.s3_client = boto3.client(
                "s3",
                region_name=region_name
            )

        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
