import copy
import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from spatial.geometry import Polygon, PolygonCollection, DynamicObject
from spatial.logic import Spatial


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


def cut_sequence(objects: dict, start, end) -> (list, dict):
    res = dict()
    ids = set()
    for i in np.arange(start, end):
        res[i] = objects[i]
        for o in objects[i].keys():
            ids.add(o)

    return list(ids), res


def setup_per_object(objects_in_time: dict, ego: int, ids: list) -> (DynamicObject, DynamicObject, list):
    other_ids = set(ids) - set([ego])
    seen = False

    ego_obj = DynamicObject()
    other_obj = DynamicObject()
    validity = list()
    times = list(objects_in_time.keys())

    for t in times:
        objs = objects_in_time[t]
        obj_keys = list(objs.keys())
        if ego in obj_keys:
            # now we have seen ego for the first time
            if not seen:
                seen = True
                validity.append(t)

            ego_obj.addObject(PolygonCollection(set([Polygon(objs[ego])])), t)
            id_filtered = set(obj_keys).intersection(other_ids)
            other_obj.addObject(PolygonCollection(set([Polygon(objs[i]) for i in id_filtered])), t)
        else:
            # if the ego is not in the list anymore, we can break
            if seen:
                validity.append(t - 1)
                break

    if len(validity) == 1:
        # in case the object is visible during the last time step
        validity.append(times[-1])

    return ego_obj, other_obj, validity


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

    # extract data of evaluation interval
    t_s = 8400
    t_d = 1800

    # sort objects after time
    time_objects = sort_objects_after_time(tracking)
    ids, seq = cut_sequence(time_objects, t_s, t_s + t_d + 1)
    # ids = ids[:15]
    print('Found the following IDs in the sequence: {}'.format(ids))

    # setup spec and SpaTiaL
    close = "(ego dist others <= 15)"
    spec = f"(G {close} ->t (G[30,60] (not{close}) ) )"
    # spec = f"({close} ->t (G[90,180] (not{close}) ) )"
    # spec = f"({close} ->t (F[0,60] (not{close}) ) )"
    spatial = Spatial(quantitative=True)
    parsed = spatial.parse(spec)

    # compute satisfaction for all objects
    satisfaction_per_object = dict()
    min_s = np.inf
    max_s = -np.inf
    for i, id in enumerate(ids):
        # satisfaction per time
        sat_per_time = dict()
        # setup objects
        ego, others, validity = setup_per_object(seq, id, ids)
        spatial.assign_variable('ego', ego)
        spatial.assign_variable('others', others)
        spatial.reset_spatial_dict()
        for t in np.arange(validity[0], validity[1] + 1):
            # evaluate spec
            s = spatial.interpret(parsed, validity[0], t)
            sat_per_time[t] = s
            if s > max_s:
                max_s = s
            if s < min_s:
                min_s = s
        satisfaction_per_object[id] = copy.deepcopy(sat_per_time)
        print('Remaining data to process: {}'.format(len(ids) - i - 1))

    print('Satisfaction range across the different objects: {} to {}'.format(min_s, max_s))
    # setup scaler
    max_v = 30
    scaler = lambda x: 1 if x > max_v else (
                x / 2 / max_v + 0.5) if -max_v <= x <= max_v else 0  # lambda x: x/np.max([min_s,max_s])/2+0.5 #MinMaxScaler(feature_range=(-np.max([min_s,max_s]),np.max([min_s,max_s])))


    def get_color(val):
        thresh = 30
        alpha_thresh = 0.5
        color = None
        fac = np.min([abs(val) / thresh, 1])
        if val >= 0 or np.isclose(val, 0):
            color = (255 - 55 * (1 - fac), 0, 0)  # , fac)
        else:
            color = (0, 0, 255 - 55 * (1 - fac))  # , fac)
        return color


    viridis = cm.get_cmap('viridis')  # , 8)

    # make video of the boxes
    cap = cv2.VideoCapture(video_file)
    time_var = 0
    print('Minimum time is {}'.format(np.min(list(seq.keys()))))
    # save to folder?
    SAVE = True
    if SAVE:
        path = os.getcwd() + str(int(time.time())) + "/"
        os.mkdir(path)

    RUNNING = True
    while (cap.isOpened() and RUNNING):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if t_s <= time_var <= t_s + t_d:
            # plot all bounding boxes
            loc_ids = set(time_objects[time_var]).intersection(set(ids))
            for o in loc_ids:
                sat = satisfaction_per_object[o][time_var]
                color = get_color(
                    sat)  # (0,0,0) if sat < 0 else (255,255,255)#np.array(viridis(scaler(sat)))[:3]*255 #(0,0,0) if sat < 0 else (255,255,255)
                vertices = time_objects[time_var][o]
                center = np.mean(vertices, axis=0)
                for idx in range(len(vertices) - 1):
                    # cv2.line(image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                    first = tuple(vertices[idx, :].astype(int))  # tuple((tag.center + (tag.corners[idx - 1, :] - tag.center) * factor).astype(int))
                    sec = tuple(vertices[idx + 1, :].astype(int))  # tuple((tag.center + (tag.corners[idx, :] - tag.center) * factor).astype(int))
                    cv2.line(frame, first, sec, color, 3)
                    # add id
                cv2.putText(frame, str(o),
                            # org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                            org=(center[0].astype(int) - 20, center[1].astype(int) + 15),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2.,
                            thickness=3,
                            color=(255, 255, 255))  # (int(np.random.random(1)*255), int(np.random.random(1)*255), int(np.random.random(1)*255)))

            cv2.imshow('Detected tags', frame)  # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if SAVE:
                cv2.imwrite(path + str(time_var).zfill(6) + ".jpg", frame)
            k = cv2.waitKey(33)

        # break loop when over
        if time_var > t_s + t_d:
            RUNNING = False
        time_var += 1

    # save data
    saved = {'t_s': t_s, 't_d': t_d, 'tobj': time_objects, 'sat': satisfaction_per_object, 'spec': spec}
    pickle.dump(saved, open(path + "data.pkl", 'wb'))

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
