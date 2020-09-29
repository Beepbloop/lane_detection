from Utilities import *

def clean_lane(dots):
    start = bool(dots[0][0] - dots[1][0] > 0)

    for i in range(1, len(dots)):
        if bool(dots[i-1][0] - dots[i][0] > 0) != start:
            dots = dots[:i]
            break
    return dots

video_num = '0250'
vid = cv.VideoCapture(get_video(video_num))

factor = 2
start_frame = 500
for i in range(1, start_frame):
    sucess, temp = vid.read()
    if not sucess:
        exit()

for i in range(start_frame, 700):
    sucess, orignal_frame = vid.read()
    if not sucess:
        print(i)
        break
    if i == 1:
        continue
    img = orignal_frame.copy()
    img = cv.resize(img, (0, 0), fx=1 / factor, fy=1 / factor)
    lanes = get_lanes(video_num, str(i), factor)
    cleaned_lanes = []
    for i in range(len(lanes)):
        cleaned_lanes.append(clean_lane(lanes[i]))
    # lanes = clean_lane(lanes)
    orignal = draw_lanes(img, lanes)
    cleaned = draw_lanes(img, cleaned_lanes)

    cv.imshow('Video_{}_orignal'.format(video_num), orignal)
    cv.imshow('Video_{}_cleaned'.format(video_num), cleaned)
    cv.waitKey(0)
vid.release()