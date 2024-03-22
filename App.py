import cv2
import mediapipe as mp
import argparse

        
args=argparse.ArgumentParser()

args.add_argument("--mode",default="video")
args.add_argument("--filePath",default="./video.mp4")
args=args.parse_args()


mp_face_detection=mp.solutions.face_detection

def blurred_files(img,face_detection):
     H,W,_=img.shape
     img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
     out=face_detection.process(img_rgb)

     if out.detections is not None:
        for detection in out.detections:
            location_data=detection.location_data
            bbox=location_data.relative_bounding_box
            x1,y1,w,h=bbox.xmin,bbox.ymin,bbox.width,bbox.height
           

            x1=int(x1*W)
            y1=int(y1*H)
            w=int(w*W)
            h=int(h*H)

            img[y1:y1+h,x1:x1+w]=cv2.blur(img[y1:y1+h,x1:x1+w],(150,150))
     return img



with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
     if args.mode in ["image"]:
        img=cv2.imread(args.filePath)
        img=blurred_files(img,face_detection=face_detection)
        cv2.imwrite("./output.png",img)
     elif args.mode in ["video"]:
        capture=cv2.VideoCapture(args.filePath)
        ret,frame=capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        output_video=cv2.VideoWriter("./output.mp4",
                                     cv2.VideoWriter_fourcc(*'MPV4'),
                                     fps,
                                     (frame.shape[1],frame.shape[0]))
        
        while ret:
            frame=blurred_files(frame,face_detection=face_detection)
            output_video.write(frame)
            ret,frame=capture.read()


        capture.release()
        output_video.release()




