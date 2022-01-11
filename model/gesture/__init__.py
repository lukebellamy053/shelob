import csv
import copy
import itertools
import mediapipe as mp
from hardware.camera import get_camera_instance
from model import KeyPointClassifier

mp_hands = mp.solutions.hands


class Gesture:

    mp_hands = None
    hands = None
    keypoint_classifier_labels = []
    keypoint_classifier: KeyPointClassifier = None

    def __init__(self, max_num_hands=1, use_static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        with open(
            'model/keypoint_classifier/keypoint_classifier_label.csv', 
            encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        self.keypoint_classifier = KeyPointClassifier()
        
    def calc_landmark_list(self,image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def process_landmarks(self, results, debug_image):
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                index = self.keypoint_classifier(pre_processed_landmark_list)
                return self.keypoint_classifier_labels[index], index
        else:
            return None

    def pre_process_landmark(self,landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def get_gesture(self):
        camera = get_camera_instance()
        ret, image, debug_image = camera.take_picture()
        if not ret:
            return
        
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        label = self.process_landmarks(results, debug_image)
        return results, label, debug_image