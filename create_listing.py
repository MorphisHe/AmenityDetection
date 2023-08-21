# lambda function to detect amenities from images uploaded to s3 bucket
import json
import urllib.parse
import boto3
import logging
import time
import requests
import uuid
from datetime import datetime
from botocore.exceptions import ClientError
from entity_keyphrase_extractor import ComprehendDetect

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def comprehend_detect(listing_desc):
    # extract NER, keyphrases, PII using AWS Comprehend
    comp_detect = ComprehendDetect(boto3.client('comprehend'))
    keywords = comp_detect.run_pipeline(listing_desc)
    return keywords

def detect_amenities(bucket, photo):
    # get image
    s3 = boto3.client('s3')
    image = s3.get_object(Bucket=bucket, Key=photo)
    image_bytes = image["Body"].read()

    # call inference
    model_end_point = "d2-sm-coco-serving-latest-inference"
    client = boto3.client('sagemaker-runtime')
    accept_type = "json" # "json" or "detectron2". Won't impact predictions, just different deserialization pipelines.
    content_type = 'image/jpeg'
    payload = image_bytes

    response = client.invoke_endpoint(
        EndpointName=model_end_point,
        Body=payload,
        ContentType=content_type,
        Accept=accept_type
    )
    predictions = response['Body'].read()
    predictions = json.loads(predictions)
    return predictions # List[str]

def creat_listing_id():
    listing_id = str(uuid.uuid4())
    return listing_id

def format_data(listing, label_list, keyword_list):
    
    #add listingid
    listing_id = creat_listing_id()    
    listing['listingid'] = listing_id
    
    ###NEED UPDATES BY XKJ###
    #add labels
    listing['label'] = label_list
    listing['keyword'] = keyword_list
    
    #add status
    listing['status'] = 'active'
    
    return listing


def post_to_dynamo(data):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('listing')

    try:
        with table.batch_writer() as writer:
            r = writer.put_item(Item=data)
        logger.info("Loaded data into table %s.", table.name)
        print("Item is successfully added to DynamoDB table %s.", table.name)
    except ClientError:
        logger.exception("Couldn't load data into table %s.", table.name)
        raise
    
    return r

def edit_dynamo(listingid, status, price):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('listing')
    
    if status != "":          
        try:
            r = table.update_item(
                    Key={'listingid': listingid},
                    UpdateExpression="set #st=:s",
                    ExpressionAttributeValues={":s": status
                                              },
                    ExpressionAttributeNames={"#st": "status",
                                             },
                    ReturnValues="UPDATED_NEW")
            logger.info("Loaded data into table %s.", table.name)
            print("Status is successfully updated to DynamoDB table %s.", table.name)
        except ClientError:
            logger.exception("Couldn't load data into table %s.", table.name)
            raise
        r = change_ES(listingid, "status", status)
    if price != "":
        try:
            r = table.update_item(
                    Key={'listingid': listingid},
                    UpdateExpression="set #pr=:p",
                    ExpressionAttributeValues={
                                              ":p": price},
                    ExpressionAttributeNames={
                                             "#pr": "price"},
                    ReturnValues="UPDATED_NEW")
            logger.info("Loaded data into table %s.", table.name)
            print("Status is successfully updated to DynamoDB table %s.", table.name)
        except ClientError:
            logger.exception("Couldn't load data into table %s.", table.name)
            raise
        r = change_ES(listingid, "price", int(price))
    
    return r
    

def create_json_es(formated_data):
        
    #form json file    
    data_es = {
        'listingid': formated_data['listingid'],
        'address': formated_data['address'],
        'area': int(formated_data['area']),
        'label': formated_data['label'],
        'layout': formated_data['layout'],
        'price': int(formated_data['price']),
        'userid': formated_data['userid'],
        'keyword': formated_data['keyword'],
        'status': formated_data['status']
    }
    
    return json.dumps(data_es)

def upload_ES(data_es):
    
    region = 'us-east-1'
    service = 'es'
    host = 'https://search-cc-proj-listing-3dp4nkig2xy7idbhpyl3tfunwu.us-east-1.es.amazonaws.com' 
    index = 'listing'
    typ = 'listing'
    eid = json.loads(data_es)['listingid']
    url = host + '/' + index + '/' + typ + '/' + eid
    
    headers = { "Content-Type": "application/json" }
    
#     client = boto3.client('es')
    payload = json.loads(data_es)
    r = requests.post(url, auth=('masteruser', '@Masteruser1'), json=payload)
    print(r.text)
    
    return r

def remove_ES(listingid):
    
    region = 'us-east-1'
    service = 'es'
    host = 'https://search-cc-proj-listing-3dp4nkig2xy7idbhpyl3tfunwu.us-east-1.es.amazonaws.com' 
    index = 'listing'
    typ = 'listing'
    eid = listingid
    url = host + '/' + index + '/' + typ + '/' + eid

    r = requests.delete(url, auth=('masteruser', '@Masteruser1'))
    print(r.text)
    
    return r

def change_ES(listingid, item, value):
    
    region = 'us-east-1'
    service = 'es'
    host = 'https://search-cc-proj-listing-3dp4nkig2xy7idbhpyl3tfunwu.us-east-1.es.amazonaws.com' 
    index = 'listing'
    typ = '_update'
    eid = listingid
    url = host + '/' + index + '/' + typ + '/' + eid
    
    headers = { "Content-Type": "application/json" }
    payload = {
                "doc": {
                    item: value
                  }
                }
    r = requests.post(url, auth=('masteruser', '@Masteruser1'), json=payload, headers =headers )
    print(r.text)
    
    return r
    
def get_listings_by_userid(userid):
    
    region = 'us-east-1'
    service = 'es'
    host = 'https://search-cc-proj-listing-3dp4nkig2xy7idbhpyl3tfunwu.us-east-1.es.amazonaws.com' 
    index = 'listing'
    typ = '_search'
    url = host + '/' + index + '/' + typ 
    
    #query userid
    query = {
        "size": 1000,
        "query": {
            "query_string": {
                "query": userid,
                "fields": ["userid"]
            }
        }
    }
    
    headers = { "Content-Type": "application/json" }
    response = requests.get(url, auth=('masteruser', '@Masteruser1'), data=json.dumps(query), headers=headers)
    
    listing_list = json.loads(response.text)['hits']['hits']
    listingid_list =[]
    for item in listing_list:
        listingid_list.append(item['_source']['listingid'])
    
    return listingid_list

def get_listing_details(listingid_list):
    
    dyn_resource = boto3.resource('dynamodb')
    table = dyn_resource.Table('listing')
    
    result_list=[]
    for listingid in listingid_list:
        item = table.get_item(Key={
                            'listingid': listingid
                            })
        result_list.append(item['Item'])
        
    
    return result_list









def lambda_handler(event, context):
    
    if event['httpMethod'] == 'GET':
        
        listingid_list = get_listings_by_userid(event['userid'])
        listing_wdetials_list =get_listing_details(listingid_list)
        
        return {'body': listing_wdetials_list}     
    
    if event['httpMethod'] == 'POST':
        
        logger.debug('{}'.format(event))

        listing = event['messages'][0]['unstructured']

        # get keywords
        keyword_list = comprehend_detect(listing["description"])

        # detect amenities from listing images
        bucket = 'cc-proj-imagebucket'
        label_list = []
        for item in listing['imgurl']:
            ###TO BE UPDATED BY XKJ###
            label_list += detect_amenities(bucket, item.split('/')[-1])

        #put record to dynamodb   
        formatted_data = format_data(listing, label_list, keyword_list) ###TO BE UPDATED BY XKJ###
        res1 = post_to_dynamo(formatted_data)
        
        #put record to elastic search
        data_es = create_json_es(formatted_data)
        res2 = upload_ES(data_es)
        
    if event['httpMethod'] == 'PUT':
        
        #change status in dynamodb/es
        res1 = edit_dynamo(event['listingid'], event['status'], event['price'])
        
    return {
        "statusCode": 200,
        'body': "success returned from lf1!"
        
    }