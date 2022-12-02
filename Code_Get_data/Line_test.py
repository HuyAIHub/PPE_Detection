### importing required libraries
import torch
import cv2
from datetime import datetime
'''
    dict_check = {1: ['Work pass', [True, True, True]],
                  2: ['Supervisor pass', [True, True, True]],
                  3: ['No helmet and shirt', [False, False, True]],
                  4: ['No helmet and shoes', [False, True, False]],
                  5: ['No Shirt and shoes', [True, False, False]],
                  6: ['No helmet', [False, True, True]],
                  7: ['No shirt', [True, False, True]],
                  8: ['No shoes', [True, True, False]],
                  9: ['Stranger', [False, False, False]]
                  }
'''

def Detectx(frame, model):
    frame = [frame]
    results = model(frame)

    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, coordinates

def Detect_Person_Croped(frame,frame_person,model,cor):
    """
        0 -> Helmet Supervisor
        1 -> Shirt
        2 -> Shoes
        3 -> Helmet Worker
        4 -> People
    """
    dict_check = {1: ['Work pass', [True, True, True]],
                  2: ['Supervisor pass', [True, True, True]],
                  3: ['No helmet and shirt', [False, False, True]],
                  4: ['No helmet and shoes', [False, True, False]],
                  5: ['No Shirt and shoes', [True, False, False]],
                  6: ['No helmet', [False, True, True]],
                  7: ['No shirt', [True, False, True]],
                  8: ['No shoes', [True, True, False]],
                  9: ['Stranger', [False, False, False]]
                  }
    labels, cord = Detectx(frame_person, model=model) # detect PPE khi đã cắt ng
    n = len(labels)
    x1, y1, x2, y2 = cor[0],cor[1],cor[2],cor[3]
    object = []
    front_scale = 0.5
    think_ness = 2
    Red = (0, 0, 255)
    Green = (29, 186, 26)

    for i in range(n):
        object.append(int(labels[i].item()))

    if 3 in object and 1 in object and 2 in object:
        cv2.putText(frame, 'W->PASS', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Green, think_ness)
        object.clear()
        return dict_check[1]
    elif 0 in object and 1 in object and 2 in object:
        cv2.putText(frame, 'S->PASS', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Green, think_ness)
        object.clear()
        return dict_check[2]
    elif 3 not in object and 0 not in object and 1 not in object:
        cv2.putText(frame, 'No helmet and shirt', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Red,think_ness)
        object.clear()
        return dict_check[3]
    elif 3 not in object and 0 not in object and 2 not in object:
        cv2.putText(frame, 'No helmet and shoes', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Red,think_ness)
        object.clear()
        return dict_check[4]
    elif 1 not in object and 2 not in object:
        cv2.putText(frame, 'No Shirt and Shoes', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Red,
                    think_ness)
        object.clear()
        return dict_check[5]
    elif 3 not in object and 0 not in object:
        cv2.putText(frame, 'No helmet', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Red, think_ness)
        object.clear()
        return dict_check[6]
    elif 1 not in object:
        cv2.putText(frame, 'No Shirt', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Red, think_ness)
        object.clear()
        return dict_check[7]
    elif 2 not in object:
        cv2.putText(frame, 'No Shoes', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Red, think_ness)
        object.clear()
        return dict_check[8]
    # else:
    #     cv2.putText(frame, 'Stranger', (x1-10, y1-10), cv2.FONT_HERSHEY_DUPLEX, front_scale, Red, think_ness)
    #     object.clear()
    #     return dict_check[9]
def plot_boxes(results, frame, cor):
    confidence_threshold = 0.6
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    x1, y1, x2, y2 = cor[0], cor[1], cor[2], cor[3]
    list_label = ['Helmet Supervisor','Vest','Shoes','Helmet Worker','People']
    COLOR = (255, 255, 255)
    label = ''
    for i in range(n):
        row = cord[i]
        if int(labels[i].item()) == 0:
            label = list_label[0]
            COLOR = (238, 245, 237)
        elif int(labels[i].item()) == 1:
            label = list_label[1]
            COLOR = (11, 30, 230)
        elif int(labels[i].item()) == 2:
            label = list_label[2]
            COLOR = (189, 15, 212)
        elif int(labels[i].item()) == 3:
            label = list_label[3]
            COLOR = (12, 114, 168)
        elif int(labels[i].item()) == 4:
            label = list_label[4]
            COLOR = (29, 186, 26)

        if row[4] >= confidence_threshold:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)  ## BBox
            cv2.putText(frame, label + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,COLOR, 2)

    return frame
def Get_person(results, frame,line,model):

    confidence_threshold = 0.1
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    COLOR = (29, 186, 26)
    OFFSET = 20
    for i in range(n):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        row = cord[i]
        if row[4] >= confidence_threshold and int(labels[i].item()) == 4:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            center = Center_handle(x1, y1, x2, y2)
            print("center point: ",center)
            if center[-1] <= line + OFFSET and  center[-1] >= line - OFFSET:
                # Img_frame = '/home/aitraining/workspace/huydq46/PPE_project_test/Crop_frame/' + current_time + '.jpg'
                # cv2.imwrite(Img_frame, frame)
                coordinate = [x1, y1, x2, y2]
                Img_copy = frame.copy()
                Crop = Img_copy[y1-10:y2 + 10, x1-10:x2+10]  # cut person
                print(Img_copy[y1-10:y2 + 10, x1-10:x2+10])
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
                out = Detect_Person_Croped(frame,Crop,model,coordinate)
                print(out)

                # Img_person = '/home/aitraining/workspace/huydq46/PPE_project_test/Crop_people/' + current_time + '.jpg'
                # cv2.imwrite(Img_person, Crop)
                print("oke")
    return frame


def Center_handle(x, y, w, h):
    cx = (x + w)//2
    cy = (y + h)//2
    return cx, cy


def main(vid_path=None):
    model = torch.hub.load('/home/huydq/PycharmProjects/yolov5_PPE', 'custom',
                           path="/home/huydq/PycharmProjects/yolov5_PPE/weight/best2.pt", source='local')
    # ## reading the video
    # cap = cv2.VideoCapture(vid_path)
    # ok, frame = cap.read()
    # frame.flags.writeable = True
    # frame = cv2.resize(frame, dsize=None, fx=0.9, fy=0.9)
    # bbox = (780, 174, 775, 798)
    # x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    LINE = 350
    # while True:
    #     try:
    #         ret, frame = cap.read()
    #         x_shape, y_shape = frame.shape[1], frame.shape[0]
    #         frame.flags.writeable = True
    #     except:
    #         print("check")
    #         cap = cv2.VideoCapture(vid_path)
    #         continue
    #     frame = cv2.resize(frame, dsize=None, fx=0.9, fy=0.9)
    #     roi_frame = frame[y1:y1 + y2, x1:x1 + x2]
    #     if ret:
    #         frame1 = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    #         results = Detectx(frame1, model=model)
    #         frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
    #         frame1 = Get_person(results,frame1,LINE,model)
    #         cv2.line(frame1, (20, LINE), (x_shape + y_shape, LINE), (0, 255, 255),
    #                  2)  # set_rule_line
    #         #cv2.imshow("vid_out", frame1)
    #     else:
    #         cap = cv2.VideoCapture(vid_path)
    #     if cv2.waitKey(5) & 0xFF == 27:
    #         break
    frame = cv2.imread('2022-07-25_18:09:33.jpg')
    results = Detectx(frame, model=model)
    Get_person(results,frame,LINE,model)
    cv2.imshow("vid_out", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()


# main(vid_path="rtsp://admin:Vcc12345678@vcc-hoangmai3.ddns.net:1571/cam/realmonitor?channel=1&subtype=0")
main()
# y = 490
if __name__ == '__main__':
    main()
