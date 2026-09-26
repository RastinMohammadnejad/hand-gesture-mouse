import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open the camera.")
        return

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read frame from camera.")
            break

        cv2.imshow("Hand Gesture Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()