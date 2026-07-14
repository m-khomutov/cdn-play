import sys
import cv2
import argparse

if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='flv[cdn] debugger')
    parser.add_argument('file', type=str, help='FLV file to watch')
    parser.add_argument('--step', action='store_true', help='show in step mode')
    args: argparse.Namespace = parser.parse_args()

    cap = cv2.VideoCapture(args.file)
    if not cap.isOpened():
        print(f"Cannot open {args.file}")
        sys.exit(1)

    cv2.namedWindow(args.file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(args.file, frame)

        if args.step:
            if cv2.waitKey(-1) == ord('q'):
                break
        else:
            if delay := int(cap.get(cv2.CAP_PROP_FPS)) == 0:
                delay = 20
            if cv2.waitKey(700//delay) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()



