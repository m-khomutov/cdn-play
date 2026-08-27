import os
import sys
import cv2
import argparse


def store():
    ret, raw_packet = cap.retrieve()
    if ret and raw_packet is not None:
        with open('pps.264', 'ab') as of:
            of.write(b'\x00\x00\x01'+raw_packet.tobytes()[492:])#+b'\x00\x00\x00\x01')
 
if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='flv[cdn] debugger')
    parser.add_argument('file', type=str, help='FLV file to watch')
    parser.add_argument('--step', action='store_true', help='show in step mode')
    parser.add_argument('--frame', type=int, default=0, help='step to frame number')
    args: argparse.Namespace = parser.parse_args()

    file_size = os.path.getsize(args.file)

    cap = cv2.VideoCapture(args.file)
    if not cap.isOpened():
        print(f"Cannot open {args.file}")
        sys.exit(1)

    cv2.namedWindow(args.file)
    total= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    #cap.set(cv2.CAP_PROP_FORMAT, -1)
    while True:
        #grabbed = cap.grab()
        #if not grabbed:
        #    break
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f'frame: {frame_number} of {total}')

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



