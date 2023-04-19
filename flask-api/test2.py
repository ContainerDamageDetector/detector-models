# import boto3
# import os
#
# s3 = boto3.client('s3')
#
# s3 = boto3.resource(
#     service_name='s3',
#     region_name='ap-south-1',
#     aws_access_key_id='AKIA5OSYVVF3F5WPQHEY',
#     aws_secret_access_key='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'
# )
#
# for bucket in s3.buckets.all():
#     print(bucket.name)
#
#
# os.environ["AWS_DEFAULT_REGION"] = 'ap-south-1'
# os.environ["AWS_ACCESS_KEY_ID"] = 'AKIA5OSYVVF3F5WPQHEY'
# os.environ["AWS_SECRET_ACCESS_KEY"] = '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'
#
# for obj in s3.Bucket('container-damage-detector').objects.all():
#     print(obj)

import boto3
import io
from PIL import Image
from urllib.parse import urlparse
from io import BytesIO


# Parse the S3 URI
s3_uri = 's3://container-damage-detector/06.jpg'
parsed_url = urlparse(s3_uri)
bucket_name = 'container-damage-detector'
key = parsed_url.path.lstrip('/')

# Create an S3 client
s3 = boto3.client('s3')

s3_object = s3.get_object(Bucket=bucket_name, Key=key)

# Read the object data and decode it from bytes to a string
object_data = s3_object['Body'].read()

# Load the image from the decoded object data
image = Image.open(BytesIO(object_data))
print()
# Display the image
image.show()

# # Retrieve the object from S3
# response = s3.get_object(Bucket=bucket_name, Key=key)
# # Access the content of the object
# content = response['Body'].read()
#
# image = Image.open(BytesIO(response))
#
# # # Open the image using PIL
# # img = Image.open(io.BytesIO(content))
# #
# # # Display the image
# image.show()
