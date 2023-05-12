import boto3
import os

def save_data(filename):
    s3 = boto3.client('s3',
                        endpoint_url='https://storage.yandexcloud.net',
                        aws_access_key_id=os.environ['access_key'],
                        aws_secret_access_key=os.environ['secret'])

    bucket_name = 'sikoraaxd-bucket'

    with open(filename, 'rb') as f:
        s3.upload_fileobj(f, bucket_name, filename)
