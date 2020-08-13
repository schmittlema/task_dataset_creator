# Create Task Dataset
# Schmittle
# Creates a dataset of images with random start/goals and mahatten paths between them

import cv2
import numpy as np
import os
import yaml

def neighbors(curr_head, img):
    '''
    Find manhatten neighbors of curr_head in img
    '''
    neighbors = []
    change = [-1, 1]
    for dx in change:
        neighbor1 = [curr_head[0]+dx, curr_head[1]]
        neighbor2 = [curr_head[0], curr_head[1]+dx]
        for idx, neighbor in enumerate([neighbor1, neighbor2]):
            if neighbor[0] >= 0 and neighbor[0] < img.shape[0] and neighbor[1] >= 0 and neighbor[1] < img.shape[1]:
                # viable neighbor
                direction = (idx*2) + max(dx, 0) # 0 = left, 1 = right, 2 = up, 3 = down
                neighbors.append((neighbor, direction))
    return neighbors

def makes_progress(dist_x, dist_y, end, end_neighbor, img, line):
    '''
    Determines if given end_neighbor makes progress towards end and is in bounds
    '''
    no_overshoot = True
    if end[0] == end_neighbor[0] or end[1] == end_neighbor[1]:
        if line[-1][0] == end[0]:
            no_overshoot = np.sign(line[-1][1] - end[1]) == np.sign(end_neighbor[1] - end[1]) or np.sign(end_neighbor[1] - end[1]) == 0
        else:
            no_overshoot = np.sign(line[-1][0] - end[0]) == np.sign(end_neighbor[0] - end[0]) or np.sign(end_neighbor[0] - end[0]) == 0
    progress = dist_x > abs(end[0] - end_neighbor[0]) or dist_y > abs(end[1] - end_neighbor[1])
    if progress:
        inbounds = end_neighbor[0] >= 0 and end_neighbor[1] >= 0 and end_neighbor[0] < img.shape[0] and end_neighbor[1] < img.shape[1]
    if progress and inbounds:
        if len(line) >= 2:
            if line[-1][0] == end_neighbor[0]:
                no_overlap = np.sign(line[-1][1] - line[-2][1]) == np.sign(end_neighbor[1] - line[-1][1]) or np.sign(line[-1][1] - line[-2][1]) == 0
            else:
                no_overlap = np.sign(line[-1][0] - line[-2][0]) == np.sign(end_neighbor[0] - line[-1][0]) or np.sign(line[-1][0] - line[-2][0]) == 0
        else:
            no_overlap = True

        return no_overshoot and progress and inbounds and no_overlap
    return False

def draw_line(img, params):
    '''
    Draw somewhat random line from random start to random goal on image
    '''
    start = end = [0,0]
    while start[0] == end[0] and start[1] == end[1]:
        start = list(np.random.randint([0,0], img.shape))
        end = list(np.random.randint([0,0], img.shape))
    line = [start]
    curr_head = start
    while curr_head[0] != end[0] or curr_head[1] != end[1]:
        dist_x = abs(end[0] - curr_head[0])
        dist_y = abs(end[1] - curr_head[1])
        successors = []
        for neighbor in neighbors(curr_head, img):
            direction = neighbor[1]
            neighbor = neighbor[0]
            end_neighbor = curr_head
            #if dist_x > abs(end[0] - neighbor[0]) or dist_y > abs(end[1] - neighbor[1]):
            if makes_progress(dist_x, dist_y, end, neighbor, img, line):
                while not makes_progress(dist_x, dist_y, end, end_neighbor, img, line):
                    dist = np.random.randint(0, max(params['img_width'], params['img_length']))

                    # make end point in direction dist
                    end_neighbor = [curr_head[0] + ((direction<2)*((direction%2)-1*(direction%2==0)))*dist, curr_head[1] + ((direction>1)*((direction%2)-1*(direction%2==0)))*dist]
                successors.append(end_neighbor)

        if len(successors) == 0:
            print(end, line[-2:], np.sign(line[-1][0] - line[-2][0]) == 0,np.sign(line[-1][1] - line[-2][1]) == 0,neighbors(curr_head,img))
        successor = successors[np.random.choice(range(0,len(successors)))]
        line.append(successor)
        curr_head = successor

        for idx in range(0,len(line)):
            if idx != 0:
                prev_point = line[idx-1]
                point = line[idx]
                if prev_point[0] == point[0]:
                    #img[point[0], min(prev_point[1],point[1]):max(prev_point[1],point[1])] = 1
                    cv2.line(img, (point[0], min(prev_point[1], point[1])), (point[0], max(prev_point[1],point[1])), [1], params['thickness'])
                else:
                    cv2.line(img, (min(prev_point[0], point[0]), point[1]), (max(prev_point[0],point[0]), point[1]), [1], params['thickness'])
                    #img[min(prev_point[0],point[0]):max(prev_point[0],point[0]), point[1]] = 1
    return img

def make_images(params):
    '''
    Make images of dataset
    '''
    dataset = []
    for i in range(0, params['dataset_size']):
        img = np.zeros((params['img_length'], params['img_width']))
        img = draw_line(img, params)
        dataset.append(img)

    cv2.imshow('img', dataset[0])
    cv2.waitKey()
    dataset = np.array(dataset)
    return dataset

def load_params():
    '''
    Loads params from config.yaml
    '''
    src_dir = os.path.dirname(os.path.realpath(__file__))
    rel_path = '/'
    filename = 'config.yaml'
    with open(src_dir+rel_path+filename, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

if __name__ == "__main__":
    params = load_params()
    dataset = make_images(params)

    src_dir = os.path.dirname(os.path.realpath(__file__))
    rel_path = '/task_dataset/'
    filename = 'dataset.npy'
    np.save(src_dir+rel_path+filename,dataset)
    print('Complete!')

