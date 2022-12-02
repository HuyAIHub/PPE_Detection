### importing required libraries
import torch
import cv2
from datetime import datetime


def Detectx(frame, model):
    frame = [frame]
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

def Plot_boxes(results, frame,line):
    confidence_threshold = 0.6
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    list_label = 'People'
    COLOR = (29, 186, 26)
    offet = 20
    for i in range(n):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        row = cord[i]
        if row[4] >= confidence_threshold and int(labels[i].item()) == 4:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            center = Center_handle(x1, y1, x2, y2)
            if center[-1] <= line + offet and  center[-1] >= line - offet:
                Img_frame = '/home/aitraining/workspace/huydq46/PPE_project_test/Crop_frame/' + current_time + '.jpg'
                cv2.imwrite(Img_frame, frame)
                Img_copy = frame.copy()
                Crop = Img_copy[y1-10:y2 + 10, x1-20:x2+10]  # cut person

                Img_person = '/home/aitraining/workspace/huydq46/PPE_project_test/Crop_people/' + current_time + '.jpg'
                cv2.imwrite(Img_person, Crop)
                print("oke")
                continue
                # cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
                # cv2.putText(frame, list_label + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,COLOR, 2)
    return frame

def Center_handle(x, y, w, h):
    cx = (x + w)//2
    cy = (y + h)//2
    return cx, cy


def main(vid_path=None):
    model = torch.hub.load('/home/aitraining/workspace/huydq46/PPE_project','custom',path = "/home/aitraining/workspace/huydq46/PPE_project/weight/best2.pt",source ='local')

    ## reading the video
    cap = cv2.VideoCapture(vid_path)
    ok, frame = cap.read()
    frame.flags.writeable = True
    frame = cv2.resize(frame, dsize=None, fx=0.9, fy=0.9)
    bbox = (780, 174, 775, 798)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    LINE = 350
    a = 0
    while True:
        try:
            ret, frame = cap.read()
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            frame.flags.writeable = True
        except:
            print("check")
            cap = cv2.VideoCapture(vid_path)
            a += 1
            if a == 5: break
            continue
        frame = cv2.resize(frame, dsize=None, fx=0.9, fy=0.9)
        roi_frame = frame[y1:y1 + y2, x1:x1 + x2]
        if ret:  
            frame1 = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            results = Detectx(frame1, model=model)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            frame1 = Plot_boxes(results,frame1,LINE)
            cv2.line(frame1, (20, LINE), (x_shape + y_shape, LINE), (0, 255, 255),
                     2)  # set_rule_line
            #cv2.imshow("vid_out", frame1)
        else:
            cap = cv2.VideoCapture(vid_path)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


main(vid_path="rtsp://admin:Vcc12345678@vcc-hoangmai3.ddns.net:1571/cam/realmonitor?channel=1&subtype=0")
# y = 490
if __name__ == '__main__':
    main()
