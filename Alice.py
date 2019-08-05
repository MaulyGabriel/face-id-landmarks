from __future__ import print_function
from imutils.video import VideoStream
from time import sleep

import cv2
import dlib
import imutils
import numpy as np
import face_recognition as fr


class FacePredictor(object):

    def __init__(self, camera):

        self.camera = camera
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.color = (255, 255, 0)
        self.white = (255, 255, 255)
        self.resize = 480

    @staticmethod
    def load_models():

        codes = np.load('models/codes.npy')
        images = np.load('models/images.npy')
        names = np.load('models/names.npy')

        return codes, images, names

    def recognition(self):

        codes, images, names = self.load_models()

        process_this_frame = True

        camera = VideoStream(src=self.camera).start()

        sleep(1)

        while True:

            frame = camera.read()
            frame = imutils.resize(frame, width=self.resize)

            rgb_small_frame = frame[:, :, ::-1]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            process = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            process_image = process.apply(gray)
            detections = self.detector(process_image, 1)

            if process_this_frame:

                face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = fr.compare_faces(images, face_encoding)
                    name = "Not identified"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = names[first_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            for k, d in enumerate(detections):
                shape = self.predictor(process_image, d)
                for i in range(1, 68):
                    cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, self.color, thickness=-1)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), self.color, 2)

                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), self.color, cv2.FILLED)

                cv2.putText(frame, name, (left + 6, bottom - 6), self.font, 0.5, self.white, 1)

            cv2.imshow("Live stream", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':

    f = FacePredictor(camera=0)
    f.recognition()
