from dotenv import load_dotenv
import os
load_dotenv()

class Config:

    S3_ACESS_KEY = os.environ.get('S3_ACESS_KEY')
    S3_SECRET_ACCESS_KEY = os.environ.get('S3_SECRET_ACCESS_KEY')
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
    AWS_S3_REGION_NAME = os.environ.get('AWS_S3_REGION_NAME')
