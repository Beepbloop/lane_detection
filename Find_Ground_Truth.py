from Utilities import *

video_num = '0250'
vid = cv.VideoCapture(get_video(video_num))

factor = 2
for i in range(1, 10000):
    sucess, orignal_frame = vid.read()
    if not sucess:
        print(i)
        break
    if i == 1:
        continue
    img = orignal_frame.copy()
    img = cv.resize(img, (0, 0), fx=1 / factor, fy=1 / factor)
    lanes = get_lanes(video_num, str(i), factor)
    doted = draw_lanes(img, lanes)

    cv.imshow('Video_{}'.format(video_num), doted)
    cv.waitKey(0)
vid.release()