import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from kafka import KafkaProducer
import os,sys
from minio import Minio

class Const():
    DB_HOST         = '10.248.243.110'
    DB_NAME         = 'vcc_ai_events'
    DB_USER         = 'postgres'
    DB_PASS         = 'Vcc_postgres@2022'
    DB_PORT         = 5432
    KAFKA_BROKER    = '10.248.243.110:39092' #9000 kafdrop
    TOPIC_EVENT     = 'personalProtectiveEquipment'
    TOPIC_PPE       = 'PPE_Msg'
    # MINIO_ADDRESS   = 'minio.congtrinhviettel.com.vn' 
    MINIO_ADDRESS   = '10.248.243.110:9000'
    BUCKET_NAME     = 'ai-images'
    ACCESS_KEY      = 'minioadmin'
    SECRET_KEY      = 'Vcc_AI@2022'
    OBJECT          = []
    RED             = (0, 0, 255)
    GREEN           = (29, 186, 26)
    CONFIDENCE_THRED = 0.7
    OFFSET          = 10
    LABEL           = ['Helmet', 'Shoes', 'Shirt']
    EVENT           = { 0: ['Pass', [True, True, True]],
                        1: ['No helmet and shirt', [False, False, True]],
                        2: ['No helmet and shoes', [False, True, False]],
                        3: ['No Shirt and shoes', [True, False, False]],
                        4: ['No helmet', [False, True, True]],
                        5: ['No shirt', [True, False, True]],
                        6: ['No shoes', [True, True, False]],
                        7: ['NO PPE', [False, False, False]]}

var = Const()

def db_connect():
    try:
        conn = psycopg2.connect(host    =   var.DB_HOST, 
                                database=   var.DB_NAME, 
                                user    =   var.DB_USER, 
                                password=   var.DB_PASS,
                                port    =   var.DB_PORT)
        cur = conn.cursor()
    except psycopg2.DatabaseError as error:
        print("error while run",error)
    return conn, cur

def kafka_connect():
    kafka_broker    = var.KAFKA_BROKER
    topic_event     = var.TOPIC_EVENT
    topic_ppe       = var.TOPIC_PPE
    producer        = KafkaProducer(bootstrap_servers=kafka_broker)
    return topic_event,topic_ppe,producer

def minio_connect():
    minio_address   = var.MINIO_ADDRESS
    minio_address1  = 'minio.congtrinhviettel.com.vn'
    bucket_name     = var.BUCKET_NAME
    client          = Minio(
                            minio_address,
                            access_key=var.ACCESS_KEY,
                            secret_key=var.SECRET_KEY,
                            secure=False  # check ssl
                            )
    return minio_address,minio_address1, bucket_name , client