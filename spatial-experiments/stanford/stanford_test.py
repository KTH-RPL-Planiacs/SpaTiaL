import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def process_tracking(file: list) -> list:
    """
    Annotation file format:
    Each line in the annotations.txt file corresponds to an annotation. Each line contains 10+ columns, separated by spaces. The definition of these columns are:

    0   Track ID. All rows with the same ID belong to the same path.
    1   xmin. The top left x-coordinate of the bounding box.
    2   ymin. The top left y-coordinate of the bounding box.
    3   xmax. The bottom right x-coordinate of the bounding box.
    4   ymax. The bottom right y-coordinate of the bounding box.
    5   frame. The frame that this annotation represents.
    6   lost. If 1, the annotation is outside of the view screen.
    7   occluded. If 1, the annotation is occluded.
    8   generated. If 1, the annotation was automatically interpolated.
    9   label. The label for this annotation, enclosed in quotation marks. (pedestrians, bikers, skateboarders, cars, buses, and golf carts)
    :param file:
    :return:
    """

    objects = dict()
    objects['pedestrian'] = dict()
    objects['biker'] = dict()
    objects['skater'] = dict()
    objects['cart'] = dict()
    objects['bus'] = dict()
    objects['car'] = dict()

    def vertify(data: list):
        """
        Converts a line of vertices to a polygon representation in numpy
        :param data: The string line of vertices
        :return: The numpy array describing the vertices of the bounding box polygon
        """
        assert len(data) == 4
        n = [float(d) for d in data]
        return np.array([[n[0], n[1]], [n[2], n[1]], [n[2], n[3]], [n[0], n[3]], [n[0], n[1]]])

    def check_consistency(object) -> bool:
        """
        Checking time consistency, i.e., are all consecutive time steps provided?
        :param object: The object to check, represented through a dictionary
        :return: True if all time steps are available, false otherwise
        """
        time = np.array(list(object.keys()))
        time_diff = time[1:] - time[0:-1]
        return np.all(time_diff == 1)

    for line in file:
        # split line with comma
        columms = str(line).split(' ')
        assert len(columms) == 10

        # obtain data
        id = int(columms[0])
        # bounding box vertices
        vertices = vertify(columms[1:5])
        # time
        t = int(columms[5])
        # type
        typ = str(columms[-1]).replace("\"", "").replace("\n", "").lower()

        # lost?
        lost = int(columms[6] == 1)

        # save data
        if id not in objects[typ] and not lost:
            objects[typ][id] = dict()
        # add if not lost label is set
        if not lost:
            objects[typ][id][t] = vertices

    # post process objects and remove inconsistent objects
    for type in objects.keys():
        object_keys = list(objects[type].keys())
        for o in object_keys:
            if not check_consistency(objects[type][o]):
                # remove object
                print('Removing object {}'.format(o))
                objects[type].pop(o)

    return objects


def extractFrames(pathIn, pathOut) -> np.array:
    os.mkdir(pathOut)
    cap = cv2.VideoCapture(pathIn)
    count = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def sort_objects_after_time(objects) -> dict:
    time_objects = dict()

    for type in objects.keys():
        object_keys = list(objects[type].keys())
        for o in object_keys:
            for t in objects[type][o].keys():
                if t not in time_objects:
                    time_objects[t] = dict()
                time_objects[t][o] = objects[type][o][t]

    return time_objects


if __name__ == '__main__':

    # folder
    folder = os.getcwd() + '/stanford/bookstore/'
    annotations = folder + 'annotations.txt'
    scene_file = folder + 'reference.jpg'
    video_file = folder + 'video.mp4'

    print('Loading data from {}'.format(folder))

    # load image
    scene = plt.imread(scene_file)

    # load tracking file
    tracking_file = open(annotations, 'r')

    # process tracking file
    tracking = process_tracking(tracking_file)

    # print('Obtained {} obstacles from file'.format(len(tracking)))
    print('Obtained {} pedestrians from file'.format(len(tracking['pedestrian'])))
    print('Obtained {} bikers from file'.format(len(tracking['biker'])))
    print('Obtained {} cars from file'.format(len(tracking['car'])))
    print('Obtained {} carts from file'.format(len(tracking['cart'])))

    # make video of the boxes
    cap = cv2.VideoCapture(video_file)
    time = 0
    time_objects = sort_objects_after_time(tracking)
    print('Minimum time is {}'.format(np.min(list(time_objects.keys()))))
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if time % 100 == 0:
            print(time)
        # plot all bounding boxes
        for o in time_objects[time]:
            vertices = time_objects[time][o]
            for idx in range(len(vertices) - 1):
                # cv2.line(image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                first = tuple(vertices[idx, :].astype(int))  # tuple((tag.center + (tag.corners[idx - 1, :] - tag.center) * factor).astype(int))
                sec = tuple(vertices[idx + 1, :].astype(int))  # tuple((tag.center + (tag.corners[idx, :] - tag.center) * factor).astype(int))
                cv2.line(frame, first, sec, (0, 0, 0))

        cv2.imshow('Detected tags', frame)  # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1)
        time += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


    # plot image
    def center(ver) -> list:
        return np.array([np.mean(ver[:, 0]), np.mean(ver[:, 1])])


    for o in list(tracking['biker'].values()):
        plt.imshow(scene)
        for t in list(o.keys()):
            vert = o[t]
            plt.plot(center(vert)[0], center(vert)[1], 'ok')
        plt.autoscale()
        plt.show(block=True)
