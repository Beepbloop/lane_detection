import cv2 as cv
import re
import os

video_dir = os.path.join('Lane_Parameters', 'Lane_Videos')
lane_dir = os.path.join('Lane_Parameters', 'Lane_Parameters')
image_dir = os.path.join('Lane_Parameters', 'Lane_Images')

def get_video(video_num):
    return os.path.join(video_dir, 'IMG_'+ video_num + '.MOV')

def write_image(video_num, frame_num, img):
    
    img_dir = os.path.join(image_dir, str(video_num))
                           
    if not (os.path.exists(img_dir)):
        os.makedirs(img_dir)
                           
    cv.imwrite(os.path.join(img_dir, str(frame_num) + '.png'), img)
    
def get_image(video_num, frame_num):                  
    return os.path.join(image_dir, str(video_num), str(frame_num) + '.png')


def get_lanes(video_num, frame_num, factor = 1):
    
    p = re.compile('[0-9]*,[0-9]*')
    file = open(os.path.join(lane_dir, video_num, str(frame_num) + '.txt'))
    lanes = []

    for line in file:
        points = p.findall(line)
        for i, point in enumerate(points):
            points[i] = tuple(map(int, point.split(',')))
            points[i] = int(points[i][0]/factor), int(points[i][1]/factor)
        
        lanes.append(points)
    file.close()
    return lanes