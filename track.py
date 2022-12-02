from collections import deque
from pathlib import Path
import torch.backends.cudnn as cudnn
import logging
import os, math
from socket import timeout
from datetime import datetime,date
# from minio import Minio
# from kafka import KafkaProducer
# from kafka.errors import KafkaError
from cConst import Const,db_connect, kafka_connect, minio_connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import numpy as np
import cv2, io, json, psycopg2, torch, time,threading, sys
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)

var = Const()
minio_address, minio_address1, bucket_name, client = minio_connect()
topic_event,topic_ppe,producer = kafka_connect()
conn,cur = db_connect()

result_event    = var.EVENT
green           = var.GREEN
red             = var.RED
confidence_threadhold = var.CONFIDENCE_THRED
offset          = var.OFFSET
today = date.today()

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
MODEL_PPE      = torch.hub.load('/home/aitraining/workspace/huydq46/PPE_StrongSort/', 'custom',
                                path="/home/aitraining/workspace/huydq46/PPE_StrongSort/weights/best_ppe.pt",
                                source='local')
MODEL_PERSON   = torch.hub.load('/home/aitraining/workspace/huydq46/PPE_StrongSort/', 'custom',
                            path="/home/aitraining/workspace/huydq46/PPE_StrongSort/weights/best_person.pt",
                            source='local')
# STRONG_SORT    = StrongSORT(model_weights = WEIGHTS / '/home/aitraining/workspace/huydq46/PPE_StrongSort/weights/osnet_x0_25_msmt17.pt',
#                                 device = 'cuda:0',
#                                 fp16 = False,
#                                 max_dist = 0.2,
#                                 max_iou_distance = 0.7)
class RunModel(threading.Thread):
    def __init__(self, cam_dict):
        super().__init__()
        self.cameraID       = cam_dict.cameraID
        self.rtsp           = cam_dict.streaming_url
        self.coordinates    = cam_dict.coordinates
        self.construction_id= int(cam_dict.construction_id)
        self.doStop         = False
        self.memory         = {}
        self.already_counted = deque(maxlen=200)
        self.name           = "thread-RunModel--" + str(self.cameraID)
        self.threadID       = int(self.cameraID)
        self.STRONG_SORT    = StrongSORT(model_weights = WEIGHTS / '/home/aitraining/workspace/huydq46/PPE_StrongSort/weights/osnet_x0_25_msmt17.pt',
                                device = 'cuda:0',
                                fp16 = False,
                                max_dist = 0.2,
                                max_iou_distance = 0.7)
        if not threading.Thread.is_alive(self):
            self.start()


    def run(self):
        
        cap = cv2.VideoCapture(self.rtsp)
        # cap = cv2.VideoCapture('/home/aitraining/workspace/huydq46/PPE_StrongSort/20220914_091022_CAM2.mp4')
        box = self.coordinates
        x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])
        LINE = 460
        line = [(901, 645), (1508, 576)]
        error_frame = 0
        while True:
            print(self.name)
            timer = cv2.getTickCount()
            try:
                if self.doStop:
                    print("do stop")
                    break
                ret, frame1 = cap.read()
                x_shape, y_shape = frame1.shape[1], frame1.shape[0]
                frame1.flags.writeable = True
            except:
                cap = cv2.VideoCapture(self.rtsp)
                error_frame += 1
                if error_frame == 5: break
                continue
            frame1 = cv2.resize(frame1,dsize=None,fx = 0.9, fy = 0.9)
            if ret:
                result_p = MODEL_PERSON(frame1).xywh[0]
                # cv2.arrowedLine(frame1,line[0],line[1],(255,255,0),5)
                conf = result_p[:, 4]
                clas = result_p[:, 5]
                boxes = result_p[:, 0:4]
                list_bboxs = self.STRONG_SORT.update(boxes.cpu(),classes=clas.cpu(), confidences=conf.cpu(), ori_img = frame1)
                # list_bboxs = self.STRONG_SORT.update(boxes.cpu(),classes=clas.cpu(), confidences=conf.cpu(), ori_img = frame1)
                if len(list_bboxs) > 0:
                    for output in list_bboxs:
                        conf= output[-1]
                        id = output[-3]
                        cls = output[-2]
                        box = output[0:4]
                        # print('bbbbox',box)
                        midpoint = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
                        if id not in self.memory:
                            self.memory[id] = deque(maxlen=2)
                        self.memory[id].append(midpoint)
                        previous_midpoint = self.memory[id][0]
                        x1, y1, x2, y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                        # cv2.arrowedLine(frame1,line[0],line[1],(155,155,250),3)
                        # cv2.rectangle(frame1, (x1, y1), (x2, y2), (245,197,7), 2)
                        # cv2.putText(frame1,str(id), (x1, y1 - 2), 0, 1 / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                        if self._intersect(midpoint, previous_midpoint, line[0], line[1]) and (id not in self.already_counted):
                            current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
                            date = today.strftime("%b-%d-%Y")
                            x1, y1, x2, y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                            midpoint = (x1+int((x2-x1)/2) , y1+int((y2-y1)/2))
                            cv2.circle(frame1, midpoint, 4, (191, 148, 8), 2)
                            cv2.arrowedLine(frame1,line[0],line[1],(155,155,250),3)
                            cv2.rectangle(frame1, (x1, y1), (x2, y2), (245,197,7), 2)
                            
                            coordinate = [x1 - 10, y1 - 10, x2 + 10, y2 + 10]
                            # print('coordinate',coordinate)
                            Img_copy = frame1.copy()
                            Crop = Img_copy[y1 - 10:y2 + 10, x1:x2 + 10]  # cut person
                            event_result = self.Detect_ppe(frame1, Crop, MODEL_PPE, coordinate)

                            image_name = 'PPE/' + 'data_' + str(date) + '/' + str(current_time) + '_' + event_result[0] + '.jpg'
                            image_url = f'https://{minio_address1}/{bucket_name}/{image_name}'
                            retval, buffer = cv2.imencode('.jpg', frame1)
                            image_string = buffer.tobytes()
                            client.put_object(bucket_name=bucket_name, object_name=image_name,
                                                data=io.BytesIO(image_string), length=len(image_string),
                                                content_type='image/jpg')
                            print("+++++++++++++++++++Push_object-++++++++++++++++++")
                            # endregion

                            self.Push_database(event_result, self.cameraID, current_time, image_url)

                            # region push kafka
                            data_send_kafka = {'cameraID': self.cameraID,
                                                'contructionID': self.construction_id,
                                                'currenttime': current_time,
                                                'imageURL': [image_url],
                                                }
                            get_events = producer.send(topic_event, json.dumps(data_send_kafka).encode('utf-8'))
                            print("------------------------send----------------------")
                            try:
                                get_events.get(timeout=10)
                            except KafkaError as e:
                                print(e)
                                continue
                            # endregion
                            angle = self._ccw(line[0], line[1],previous_midpoint)
                            self.already_counted.append(id)  # Set already counted for ID to true.
                            if angle == True: 
                                time_in = datetime.now()
                                dt_string_in = time_in.strftime("%d:%m:%Y:%H:%M:%S")
                                self.angle_in_out = "in"
                                self.time_check = dt_string_in 
                                #endregion
                            elif angle == False:
                                time_out = datetime.now()
                                dt_string_out = time_out.strftime("%d:%m:%Y:%H:%M:%S")
                                self.angle_in_out = "out"
                                self.time_check = dt_string_out

                if len(self.memory) > 200:
                        del self.memory[list(self.memory)[0]]
                FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                print("FPS:",round(FPS))
                # cv2.imshow(self.name, frame1)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                        
        time.sleep(0.01)
        cap.release()
        cv2.destroyAllWindows()

    def _intersect(self, A, B, C, D):
            # kiem tra (A,C,D) co khac chieu voi (B,C,D)
            # kiem tra (A,B,C) co khac chieu voi (A,B,D)
            return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)

    def _ccw(self, A,B,C):
        # Neu return true (A,B,C) cung chieu kim dong ho va ngc lai
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))

    def Detect_peron(self, frame, model):
        frame = [frame]
        results = model(frame)
        labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, coordinates

    def Detect_ppe(self, frame, frame_person, model, coordinate_person,FRONT_SCALE = 0.5,THINK_NESS = 2,FRONT_TEXT = cv2.FONT_HERSHEY_DUPLEX ):
        """
            detect đồ bảo hộ trên ảnh người đã cắt ra
            0 -> helmet
            1 -> Shoes
            2 -> vest
        """
        # region detect PPE khi đã cắt ng
        results_person = self.Detect_peron(frame_person, model=model)
        # endregion
        labels, cord = results_person
        # region plot box
        self.Plot_boxes(results_person, frame, frame_person, coordinate_person)
        # endregion
        object = []
        n = len(labels)
        x1, y1, x2, y2 = coordinate_person[0], coordinate_person[1], coordinate_person[2], coordinate_person[3]
        for i in range(n):
            confidence = cord[i]
            if confidence[4] >= confidence_threadhold:
                object.append(int(labels[i].item()))
        #region event
        if 0 in object and 1 in object and 2 in object:
            cv2.putText(frame, 'W->PASS', (x1, y1), FRONT_TEXT, FRONT_SCALE, green, THINK_NESS)
            object.clear()
            return result_event[0]
        elif 0 not in object and 2 not in object:
            cv2.putText(frame, 'No helmet and shirt', (x1, y1), FRONT_TEXT, FRONT_SCALE, red, THINK_NESS)
            object.clear()
            return result_event[1]
        elif 0 not in object and 1 not in object:
            cv2.putText(frame, 'No helmet and shoes', (x1, y1), FRONT_TEXT, FRONT_SCALE, red, THINK_NESS)
            object.clear()
            return result_event[2]
        elif 1 not in object and 2 not in object:
            cv2.putText(frame, 'No Shirt and Shoes', (x1, y1), FRONT_TEXT, FRONT_SCALE, red,THINK_NESS)
            object.clear()
            return result_event[3]
        elif 0 not in object:
            cv2.putText(frame, 'No helmet', (x1, y1), FRONT_TEXT, FRONT_SCALE, red, THINK_NESS)
            object.clear()
            return result_event[4]
        elif 2 not in object:
            cv2.putText(frame, 'No Shirt', (x1, y1), FRONT_TEXT, FRONT_SCALE, red, THINK_NESS)
            object.clear()
            return result_event[5]
        elif 1 not in object:
            cv2.putText(frame, 'No Shoes', (x1, y1), FRONT_TEXT, FRONT_SCALE, red, THINK_NESS)
            object.clear()
            return result_event[6]
        elif 0 and 1 and 2 not in object:
            cv2.putText(frame, 'No PPE', (x1, y1), FRONT_TEXT, FRONT_SCALE, red, THINK_NESS)
            object.clear()
            return result_event[7]
        #endregion

    def Sql_insert(self, insert_scrip, insert_value):
        cur.execute(insert_scrip, insert_value)
        conn.commit()

    def Push_database(self, event_result, cameraID, current_time, path_img):
        Insert_event = 'INSERT INTO vcc_events_management.event(type_id, camera_id, created_at, captured_image_url) VALUES (%s, %s, %s, %s)'
        Insert_event_values = (6, cameraID, current_time, path_img)
        Insert_people_detection_event = "INSERT INTO vcc_events_management.people_detection_event(id,direction, user_id, user_type, is_stranger, is_wear_helmet, is_wear_shirt, is_wear_shoes) VALUES(%s, %s, %s, %s,%s, %s, %s, %s)"
        self.Sql_insert(Insert_event, Insert_event_values)
        get_data = 'SELECT id FROM vcc_events_management.event ORDER by id LIMIT all'
        cur.execute(get_data)
        event_id = cur.fetchall()
        Insert_people_detection_event_values = (
        event_id[-1], None, None, None, None, event_result[1][0], event_result[1][1], event_result[1][2])
        self.Sql_insert(Insert_people_detection_event, Insert_people_detection_event_values)


    def Center_handle(self, x, y, w, h):
        cx = (x + w) // 2
        cy = (y + h) // 2
        return cx, cy

    def Box_croped_to_box_base(self, box_base, box_crop):
            print(box_base, box_crop)
            x1, y1, x2, y2 = box_base[0], box_base[1], box_base[2], box_base[3]
            _x1, _y1, _x2, _y2 = box_crop['x1'], box_crop['y1'], box_crop['x2'], box_crop['y2']
            x = x1 + _x1
            y = y1 + _y1
            x_ = x1 + _x2
            y_ = y1 + _y2
            return x, y, x_, y_


    def Plot_boxes(self, results_person, frame, frame_person, coordinate_person):
        confidence_threshold = 0.7
        labels, cord = results_person
        n = len(labels)
        LABEL = ['Helmet','Shoes','Shirt']
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        x_shape_person, y_shape_person = frame_person.shape[1], frame_person.shape[0]
        LST_COLOR = {'helmet': (255, 255, 255), 'vest': (153, 255, 102), 'shoes': (255, 0, 204)}
        label = ''
        for i in range(n):
            row = cord[i]
            if int(labels[i].item()) == 0:
                label = LABEL[0]
                COLOR = LST_COLOR['helmet']
            elif int(labels[i].item()) == 1:
                label = LABEL[1]
                COLOR = LST_COLOR['shoes']
            elif int(labels[i].item()) == 2:
                label = LABEL[2]
                COLOR = LST_COLOR['vest']

            if row[4] >= confidence_threshold:
                x1, y1, x2, y2 = int(row[0] * x_shape_person), int(row[1] * y_shape_person), int(
                    row[2] * x_shape_person), int(row[3] * y_shape_person)
                cord_ = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                x1, y1, x2, y2 = self.Box_croped_to_box_base(coordinate_person, cord_)
                cv2.rectangle(frame, (x1 + 5, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, label + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,COLOR, 2)

        return frame
