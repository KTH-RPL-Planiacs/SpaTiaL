import copy
import os
import pickle
import time
from enum import Enum
from typing import List

import cv2
import numpy as np
import pupil_apriltags as ap
import pyrealsense2 as rs
from fpl.logics.pxtl import Image2D


class ALIGN(Enum):
    """
    Class representing different alignment options
    NO = no alignment
    D2C = depth to color
    C2D = color to depth
    """
    NO = 0
    D2C = 1  # DEFAULT
    C2D = 2


class L515Config(object):
    """
    Object which represents different configurations of the Intel Realsense L515
    """

    # default config
    @classmethod
    def config_default(cls):
        # TODO: DONT USE THIS CONFIGURATION -> IF DEPTH RES > COLOR RES, then camera will freeze
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        return config, True, True

    # config low res
    @classmethod
    def config_low_res(cls):
        config = rs.config()
        # config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
        return config, True, True

    # config high res
    @classmethod
    def config_high_res(cls):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        return config, True, True

        # config high res

    @classmethod
    def config_high_res_cam_only(cls):
        config = rs.config()
        # config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        return config, False, True  # config, depth enabled?, color enabled?


class L515(object):
    """
    Class representing an Intel Realsense L515 camera. This class serves as an interface to pyrealsense2.
    """

    def __init__(self, align: ALIGN = ALIGN.D2C, config=L515Config.config_low_res(), debug: bool = False):
        self.device = rs.context().query_devices()[0]
        self.name = self.device.get_info(rs.camera_info.name)
        assert "L515" in self.name

        # setup pipeline
        self.config = config

        # print debug info?
        if debug:
            rs.log_to_console(rs.log_severity.debug)
        self.pipeline = rs.pipeline()

        # self.queue = rs.frame_queue(30)#, keep_frames=True)
        self.profile = self.pipeline.start(self.config[0])  # , self.queue)
        self.depth_enabled = config[1]
        self.color_enabled = config[2]

        # get depth scale (meters per unit)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # setup intrinsic parameters
        temp = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intrinsic = [temp.fx, temp.fy, temp.ppx, temp.ppy]

        # setup alignment process
        if align == ALIGN.NO:
            self._alignment = None
        elif align == ALIGN.C2D:
            self._alignment = rs.align(rs.stream.depth)
        elif align == ALIGN.D2C:
            self._alignment = rs.align(rs.stream.color)

    @property
    def frames(self) -> (np.ndarray, np.ndarray, float):
        """
        Returns a pair of color and depth images together with a timestamp. The images are aligned if alignment has
        been set in the camera options
        :return: A tuple of (color image [np.ndarray], depth image [np.ndarray], timestamp [float])
        """
        frames = self.pipeline.wait_for_frames()

        # frames = None
        # color = np.asanyarray(self.queue.wait_for_frame().data)
        # depth = np.asanyarray(self.queue.wait_for_frame().data)
        # return color, depth, 0.

        # align streams if desired
        if self._alignment is not None:
            aligned_frames = self._alignment.process(frames)
            depth = np.asanyarray(aligned_frames.get_depth_frame().get_data()) * self.depth_scale if self.depth_enabled else None
            color = np.asanyarray(aligned_frames.get_color_frame().get_data()) if self.color_enabled else None
            if not depth or not color:
                print("Not yet aligned!")

        else:
            depth = np.asanyarray(frames.get_depth_frame().get_data()) * self.depth_scale if self.depth_enabled else None
            color = np.asanyarray(frames.get_color_frame().get_data()) if self.color_enabled else None

        return (color, depth, frames.get_timestamp())

    def intrinsic_param(self) -> np.ndarray:
        """
        Returns the intrinsic parameters of the camera
        :return: list of intrinsic parameters
        """
        return self._intrinsic

    def shutdowm(self):
        """
        Shuts down the camera pipeline
        :return:
        """
        self.pipeline.stop()

    def __str__(self):
        return self.name


class AprilTagDetector(object):
    """
    Class representing an AprilTag detector.
    """

    def __init__(self, params: list, tag_size: float, families='tagStandard41h12'):
        """
        Initializes the AprilTag detector
        :param params: The intrinsic camera parameters
        :param tag_size: The tag size in meters
        :param families: The considered AprilTag family (default is tagStandard41h12)
        """
        self.detector = ap.Detector(families=families,
                                    nthreads=2,
                                    quad_decimate=2.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

        self.params = params
        self.tag_size = tag_size

    def detect(self, img: np.ndarray) -> List[ap.Detection]:
        """
        Detects all tags in a given image
        :param img: The image to analyze
        :return: The list of detected AprilTags
        """
        return self.detector.detect(img, True, self.params, self.tag_size)

    def detect_tag(self, img: np.ndarray, tag: int):
        """
        Detects all tags with a predefined ID in a given image
        :param img: The image to analyze
        :param tag: The ID of the Tag
        :return: The list of detected AprilTags with specified tag ID
        """
        tags = self.detect(img)
        filtered = [t for t in tags if t.tag_id == tag]
        return filtered

    def visualize_tags(self, image: np.ndarray, tags: List[ap.Detection], wait: bool = True):
        tag_scale = {0: (1.8, 1.8), 1: (1.8, 1.8), 2: (1.8, 1.8), 3: (1.8, 1.8), 4: (1.8, 1.8), 5: (1.8, 1.8),
                     10: (2.8, 5.7), 11: (2.5, 6.5), 12: (3, 3.2), 13: (3, 3), 14: (3.2, 6), 15: (4.1, 7.4), 16: (8, 2.5)}
        # factor=1.8
        for tag in tags:
            # get scaling factor
            factor = np.array(tag_scale[tag.tag_id])
            for idx in range(len(tag.corners)):
                # cv2.line(image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                first = tuple((tag.center + (tag.corners[idx - 1, :] - tag.center) * factor).astype(int))
                sec = tuple((tag.center + (tag.corners[idx, :] - tag.center) * factor).astype(int))
                cv2.line(image, first, sec,
                         (0, 255, 0))

            cv2.putText(image, str(tag.tag_id),
                        # org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                        org=(tag.center[0].astype(int) - 20, tag.center[1].astype(int) + 15),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2.8,
                        thickness=3,
                        color=(255, 255, 255))

        cv2.imshow('Detected tags', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if wait:
            while cv2.waitKey(0) != 27:
                pass
            cv2.destroyAllWindows()

        return image

    def order_tags(self, pts: List[np.ndarray]):
        """
        Sorts a given list of pixel vectors according to their center's distance to the origin (0,0)
        :param pts: The list of points
        :return: The ordered list of points
        """
        sort = copy.copy(pts)
        sort.sort(key=lambda x: np.linalg.norm(x))
        return sort


def frames(pipeline, depth_scale, only_color: bool) -> (np.ndarray, np.ndarray, float):
    """
    Returns a pair of color and depth images together with a timestamp. The images are aligned if alignment has
    been set in the camera options
    :return: A tuple of (color image [np.ndarray], depth image [np.ndarray], timestamp [float])
    """
    frames = pipeline.wait_for_frames()

    if not only_color:
        ALIGNER = rs.align(rs.stream.color)

    # align streams
    if not only_color:
        aligned_frames = ALIGNER.process(frames)
    depth = frames.get_depth_frame() if not only_color else None
    color = frames.get_color_frame()

    return (
        np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()) * depth_scale if depth else None, frames.get_timestamp())


def process_blocks(block_tags: List[ap.Detection]) -> dict:
    # factor to resize blocks to get whole print
    tag_scale = {0: (1.8, 1.8), 1: (1.8, 1.8), 2: (1.8, 1.8), 3: (1.8, 1.8), 4: (1.8, 1.8), 5: (1.8, 1.8),
                 10: (2.8, 5.7), 11: (2.5, 6.5), 12: (3, 3.2), 13: (3, 3), 14: (3.2, 6), 15: (4.1, 7.4), 16: (8, 2.5)}

    # process blogs and store them in dict acc. to id
    blocks = dict()
    for b in block_tags:
        factor = np.array(tag_scale[b.tag_id])
        vertices = [b.center + (b.corners[i, :] - b.center) * factor for i in range(len(b.corners))]
        blocks[b.tag_id] = vertices

    return blocks


if __name__ == '__main__':

    ####
    # LIVE SETS LIVE OR RECORDED MODE
    ####
    LIVE = False
    REGISTRATION = False
    file = '/Users/chris/Documents/20210422_184459.bag'
    file = '/Users/chris/Documents/20210427_131005.bag'
    file = '/Users/chris/Documents/SpaTL_Exp3CoffeScene5.bag'

    # save figures + data to folder?
    TO_FILE = True

    if TO_FILE:
        directory = os.getcwd() + '/data_' + str(int(time.time())) + '/'
        os.mkdir(directory)
        print('Writing to file system active. Will write everything to {}'.format(directory))
    else:
        print('Data will not be saved to the filesystem! ')

    if LIVE:
        ####
        # SETUP INTEL REALSENSE CAMERA
        ####

        print('Using live mode!')

        # setup L515 in high res mode and disable alignment of images
        cam = L515(align=ALIGN.NO, config=L515Config.config_high_res_cam_only())
        # print camera
        print("Found camera {}".format(cam))

        intrinsic_param = cam.intrinsic_param()
    else:
        ####
        # SETUP INTEL REALSENSE PLAYER
        ####

        only_color = True

        print('Using playback mode with file = {}!'.format(file))
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(file, repeat_playback=False)
        profile = pipe.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

        # setup intrinsic parameters
        temp = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsic_param = [temp.fx, temp.fy, temp.ppx, temp.ppy]
        depth_scale = 0
        if not only_color:
            # get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

    # create april tag detector for image corners and blocks
    det_corner = AprilTagDetector(intrinsic_param, 0.03)  # , families='tag36h11')#cam.intrinsic_param(), 0.065)#1)
    det_blocks = AprilTagDetector(intrinsic_param, 0.018, families='tag36h11')  # 0.018m for small blocks
    det_blocks = AprilTagDetector(intrinsic_param, 0.038, families='tag36h11')  # 0.038m for small blocks

    ####
    # OBTAIN CORNERS OF IMAGE TO CUT UNNECESSARY SPACE
    ####

    # get registration tags
    if REGISTRATION:
        tags = list()
        print('Trying to find registration tags!')
        while not len(tags) == 4:
            # get example frame
            color, depth, age = cam.frames if LIVE else frames(pipe, depth_scale, only_color)
            tags = det_corner.detect_tag(np.uint8(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)), tag=0)
        centers = np.array(det_corner.order_tags([t.center for t in tags]))
        print('....found {} registration tags!'.format(len(centers)))
    # det_corner.visualize_tags(color, tags)

    ####
    # WHILE LOOP TO DETECT BLOCKS IN IMAGE
    ####
    i = 0
    running = True
    print('Starting block detection!')
    block_data = list()
    while (running and (True if LIVE else (playback.current_status() == rs.playback_status.playing))):
        t0 = time.time()
        # receive new frames from camera
        color, depth, age = cam.frames if LIVE else frames(pipe, depth_scale, only_color)
        # color, depth, age = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()), None, None
        color = Image2D(color)
        i += 1

        # straighten image
        if REGISTRATION:
            im2 = color.straigthen(centers)
        else:
            im2 = color
        im2_bw = Image2D(cv2.cvtColor(im2.data, cv2.COLOR_BGR2GRAY))  # im2.rgb2gray()

        # detect blocks
        block_tags = det_blocks.detect(np.uint8(im2_bw.data))

        # visualize blocks
        image_w_labels = det_blocks.visualize_tags(np.uint8(im2.data), block_tags, wait=False)

        # process blocks (i.e., obtain vertices)
        blocks = process_blocks(block_tags)

        if TO_FILE:
            block_data.append(blocks)
            file_name = directory + str(i).zfill(6) + '.jpg'
            # flag = cv2.imwrite(file_name, image_w_labels)
            flag = cv2.imwrite(file_name, cv2.cvtColor(image_w_labels, cv2.COLOR_RGB2BGR))
            print('Writing to file = {} is {}'.format(file_name, flag))

        t1 = time.time() - t0
        print('Processing took {}ms'.format(t1 * 1000))

        # wait for stop signal
        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            print('Shutting down...')
            cv2.destroyAllWindows()
            if LIVE:
                cam.shutdowm()
            running = False
    if TO_FILE:
        pickle.dump(block_data, open(directory + 'polygons.pkl', 'wb'))
    print('... finished!')
    exit(0)
