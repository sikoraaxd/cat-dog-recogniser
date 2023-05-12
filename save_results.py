import boto3

def save_data(filename):
    s3 = boto3.client('s3',
                  endpoint_url='https://storage.yandexcloud.net',
                  aws_access_key_id='YCAJEffTHMPCfFn4jBYUDB6oV',
                  aws_secret_access_key='YCNdxgyAtr7bUzU0iIQeQi9ViIJ7GS-ZdbiR3Fyo')

    bucket_name = 'sikoraaxd-bucket'

    with open(filename, 'rb') as f:
        s3.upload_fileobj(f, bucket_name, filename)
