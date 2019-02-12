# extractor.py
import cv2
import time

# interval if specified should be in terms of seconds


def extract_frames(detector, input_file, class_labels, interval=None, dest_dir='.'):
    print('Detection Algorithm:%s\nModel loaded: %s\nInput File: %s\nIntervals at which to detect: %r seconds' % (
        detector.name, detector.model_name, input_file, interval))
    interval = interval * 1000  # convert to milliseconds
    cap = cv2.VideoCapture(input_file)
    output = {}  # format: File name : [list of matched labels]
    if (cap.isOpened() == False):
        print("Error opening video file", input_file)
    i, count = 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_file_name = '.'.join(input_file.split('.')[:-1])
    total_duration = 0

    if dest_dir[-1] != '/':
        dest_dir = dest_dir + '/'

    while(cap.isOpened()):
        opfname = None
        if interval:
            cap.set(cv2.CAP_PROP_POS_MSEC, interval)
            opfname = input_file_name+'-'+str(int(interval/1000))+'s.jpg'
            interval += interval
        ret, frame = cap.read()
        i += 1
        if ret == True:
            print('Frame: ', i, end=' ')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            opfname = opfname if opfname else input_file_name + \
                '-'+str(int(i/fps))+':'+str(i % fps)+'.jpg'

            t1 = time.time()
            result = detector.detect(
                frame, dest_dir+opfname, class_labels)
            t2 = time.time()
            count += 1
            duration = t2-t1
            total_duration += duration
            if result:
                result.update({'time': duration})
                output.update({opfname: result})
        else:
            break

    if i and count:
        print('Overall Processing speed per image', (total_duration)/i)
        print('Frames with detections:%r/%r' % (count, i))
    else:
        print('No frames were processed')
    cap.release()
    cv2.destroyAllWindows()
    return output
