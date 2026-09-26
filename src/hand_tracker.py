import cv2
import mediapipe as mp
import pyautogui


MODEL_PATH = "models/hand_landmarker.task"


def main():
    screen_width, screen_height = pyautogui.size()

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open the camera.")
        hand_landmarker.close()
        return

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = hand_landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                index_finger = hand_landmarks[8]

                index_x = int(index_finger.x * frame_width)
                index_y = int(index_finger.y * frame_height)

                screen_x = int(index_finger.x * screen_width)
                screen_y = int(index_finger.y * screen_height)

                pyautogui.moveTo(screen_x, screen_y)

                cv2.circle(
                    frame,
                    (index_x, index_y),
                    10,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Screen: ({screen_x}, {screen_y})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                for landmark in hand_landmarks:
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1,
                    )

                for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    start = hand_landmarks[connection.start]
                    end = hand_landmarks[connection.end]

                    start_point = (
                        int(start.x * frame_width),
                        int(start.y * frame_height),
                    )

                    end_point = (
                        int(end.x * frame_width),
                        int(end.y * frame_height),
                    )

                    cv2.line(
                        frame,
                        start_point,
                        end_point,
                        (255, 0, 0),
                        2,
                    )

        cv2.imshow("Hand Gesture Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_landmarker.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()