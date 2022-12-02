### importing required libraries
import torch
import cv2
from datetime import datetime



def detectx(frame, model):
    frame = [frame]
    results = model(frame)
    #results.show()

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy
    
def main(vid_path=None):
    
    device = '1' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('/home/aitraining/workspace/huydq46/PPE_project','custom',path = "/home/aitraining/workspace/huydq46/PPE_project/weight/best2.pt",source ='local')
    classes = model.names

    ## reading the video
    cap = cv2.VideoCapture(vid_path)
    ok, frame = cap.read()
    frame.flags.writeable = True
    frame = cv2.resize(frame, dsize=None, fx=0.9, fy=0.9)
    # bbox = cv2.selectROI('Tracking', frame, False)
    bbox = (780, 174, 775, 798)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    print(bbox)
    # cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
    n = 30641
    up = 0
    while True:
        ret, frame = cap.read()
        frame.flags.writeable = True
        frame = cv2.resize(frame, dsize=None, fx=0.9, fy=0.9)
        roi_frame = frame[y1:y1+y2,x1:x1+x2]
        if ret and up % 10 == 0:
            frame1 = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            results = detectx(frame1, model=model)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            labels, cord = results
            print(labels)
            if len(labels) > 3:
                img_path = '/home/aitraining/workspace/huydq46/PPE_project/Cap_data1/' + str(n) + ".jpg"
                cv2.imwrite(img_path, frame1)
                n += 1
                print("n = ",n)
            # cv2.imshow("vid_out", frame1)
            
        if cv2.waitKey(5) & 0xFF == 27:
            break
        up += 1
    # out.release()
    cap.release()
    cv2.destroyAllWindows()


main(vid_path="rtsp://admin:Vcc12345678@vcc-hoangmai3.ddns.net:1571/cam/realmonitor?channel=1&subtype=0")
# (953, 174, 775, 798)
if __name__ == '__main__':
    main()
