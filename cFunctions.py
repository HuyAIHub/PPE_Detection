from kafka import KafkaConsumer, TopicPartition
import json
from cVariables import GlobVar
from cConst import Const
import cv2
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

var = Const()

Topic_PPE = var.TOPIC_PPE
kafka_broker = var.KAFKA_BROKER
consumer = KafkaConsumer(Topic_PPE,bootstrap_servers=kafka_broker,auto_offset_reset ='latest')

class GlobFunc():
    def __init__(self, parent =None):
        super().__init__()

    def readMessage():
        message = consumer.poll(1.0)
        if len(message.keys()) == 0:
            pass
        if TopicPartition(topic=Topic_PPE, partition=0) in message.keys():
            data = message[TopicPartition(topic=Topic_PPE, partition=0)]
            print(data)
            try:
                GlobVar.dict_cam =[]
                for _ in range(data.__len__()):
                    cam = camera()
                    cam.cameraID = json.loads(data[_].value)['cameraID']
                    cam.streaming_url = json.loads(data[_].value)['streaming_url']
                    cam.coordinates = json.loads(data[_].value)['coordinates']["personalProtectiveEquipment"][0]
                    # cam.image_url =json.loads(data[_].value)['123']
                    cam.construction_id = json.loads(data[_].value)['construction_id']
                    cam.command = json.loads(data[_].value)['cmd']
                    GlobVar.dict_cam.append(cam)
                print("mess done!")

                return True

            except Exception as e:
                # print(e)
                return False

class camera():
    cameraID = None
    streaming_url = None
    coordinates = None
    # image_url = None
    construction_id = None
    isconnected = False
    command = None
    