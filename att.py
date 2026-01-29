import cv2
import numpy as np
import os
import csv
from datetime import datetime

class FaceAttendanceSystem:
    def __init__(self):
        self.attendance_file = "attendance_log.csv"
        self.faces_dir = "faces"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = {} 
        self.is_trained = False
        
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
        
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Action", "Time"])
        
        self.load_and_train()

    def load_and_train(self):
        face_samples = []
        ids = []
        image_paths = [os.path.join(self.faces_dir, f) for f in os.listdir(self.faces_dir) if f.endswith('.jpg')]
        
        if not image_paths:
            self.is_trained = False
            return

        for image_path in image_paths:
            filename = os.path.split(image_path)[-1]
            name = filename.split('.')[0]
            
            if name not in self.names.values():
                new_id = len(self.names)
                self.names[new_id] = name
            
            current_id = [k for k, v in self.names.items() if v == name][0]
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces = self.face_cascade.detectMultiScale(img)
            
            for (x, y, w, h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(current_id)

        if face_samples:
            self.recognizer.train(face_samples, np.array(ids))
            self.is_trained = True
            print(f"System ready. {len(self.names)} users loaded.")

    def register_user(self, frame, name):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) != 1:
            return "Ensure exactly one face is visible."
            
        for (x, y, w, h) in faces:
            cv2.imwrite(f"{self.faces_dir}/{name}.jpg", frame)
            return f"User {name} registered."

    def mark_attendance(self, name):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        last_action = "Punch-out"
        with open(self.attendance_file, 'r') as f:
            rows = list(csv.reader(f))
            user_rows = [r for r in rows if r[0] == name]
            if user_rows:
                last_action = user_rows[-1][1]

        new_action = "Punch-out" if last_action == "Punch-in" else "Punch-in"
        
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, new_action, timestamp])
        
        return f"{new_action}: {name} at {timestamp}"

    def run(self):
        cam = cv2.VideoCapture(0)
        print("Controls: 'r' Register, 'p' Punch, 'q' Quit")

        while True:
            ret, frame = cam.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            current_name = "Unknown"

            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                
                if self.is_trained:
                    id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
                    if confidence < 80: 
                        current_name = self.names.get(id, "Unknown")
                        cv2.putText(frame, f"{current_name}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow('Attendance System', frame) 

            k = cv2.waitKey(10) & 0xff
            if k == ord('q'):
                break
            elif k == ord('r'):
                u_name = input("Enter name: ")
                print(self.register_user(frame, u_name))
                self.load_and_train()
            elif k == ord('p'):
                if current_name != "Unknown":
                    print(self.mark_attendance(current_name))

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceAttendanceSystem()
    app.run()