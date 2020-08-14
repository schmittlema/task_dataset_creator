import cv2

def decompress(line, width, length):
    '''
    Decompresses line points into an image
    '''
    img = np.zeros((length, width))
    for idx in range(0,len(line)):
        if idx != 0:
            prev_point = line[idx-1]
            point = line[idx]
            if prev_point[0] == point[0]:
                cv2.line(img, (point[0], min(prev_point[1], point[1])), (point[0], max(prev_point[1],point[1])), [1], params['thickness'])
            else:
                cv2.line(img, (min(prev_point[0], point[0]), point[1]), (max(prev_point[0],point[0]), point[1]), [1], params['thickness'])
    return img
