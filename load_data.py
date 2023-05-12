import os
import boto3

if __name__ == '__main__':
      s3 = boto3.client('s3',
                        endpoint_url='https://storage.yandexcloud.net',
                        aws_access_key_id=os.environ['access_key'],
                        aws_secret_access_key=os.environ['secret'])

      bucket_name = 'sikoraaxd-bucket'

      response = s3.get_object(Bucket=bucket_name, Key='dataset.rar')
      with open('./dataset.rar', 'wb') as f:
          f.write(response['Body'].read())
