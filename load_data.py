import zipfile
import boto3

if __name__ == '__main__':
      s3 = boto3.client('s3',
                        endpoint_url='https://storage.yandexcloud.net',
                        aws_access_key_id='YCAJEffTHMPCfFn4jBYUDB6oV',
                        aws_secret_access_key='YCNdxgyAtr7bUzU0iIQeQi9ViIJ7GS-ZdbiR3Fyo')

      bucket_name = 'sikoraaxd-bucket'

      response = s3.get_object(Bucket='my-bucket', Key='dataset.rar')
      with open('./dataset.rar', 'wb') as f:
          f.write(response['Body'].read())
      with zipfile.ZipFile('./dataset.rar', 'r') as zip_ref:
            zip_ref.extractall('.')