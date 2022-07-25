# standard library imports
import os
import mimetypes
import subprocess
import sys
import shutil
import sqlite3
from sqlite3 import Error
from math import sqrt
# import time

# external modules imports
import wget
import git
import cv2
from send2trash import send2trash
# import concurrent.futures
import ffmpeg
import numpy as np


def check_files(file_path):
    """
        Checks if the file to examine exists, if it's a video and if it is an mp4 file
        :param file_path: path of the file to check
        :return: True if the conditions are met, False otherwise
    """

    detect_path = '../utils/modified_detect.py'

    if not os.path.exists(file_path):
        print('The file does not exist')
        return None

    if not mimetypes.guess_type(file_path)[0].startswith('video'):
        print('The file is not a video')
        return None

    if file_path[-3:] != 'mp4':
        print('The file is not an .mp4 video, converting...', end='')
        aux_path = file_path[:-3] + 'mp4'
        if not os.path.exists(aux_path):
            command = ['ffmpeg', '-i', file_path, '-vcodec', 'copy', '-an', aux_path]
            subprocess.check_call(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print('Converted')
        else:
            print('Converted file already exists')
        file_path = aux_path

    if not os.path.exists(detect_path):
        print('Missing important files, something went wrong')
        return None

    return file_path


def check_if_db_exists(filename):
    """
        Returns true if the db corresponding to the file exists
        :param filename: path of the file to be processed
        :return: True if the database exists
    """
    return False  # test purposes
    # name = filename.split('/')[-1][:-4]
    # if os.path.exists('../db/' + name + '.db'):
    #     return True
    # return False


def get_necessary_files(small, gpu):
    """
        Gets all the necessary files for the script to work
        :param gpu: True if the user wants to use the GPU for YOLO
        :param small: True if the small model of YOLO has to be used
        :return: Void
    """

    small_link = 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt'
    xl_link = 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt'

    os.chdir('..')
    if not os.path.exists('yolov5'):
        os.mkdir('yolov5')

    if not os.path.exists('frames'):
        os.mkdir('frames')

    if not os.path.exists('rotated_frames'):
        os.mkdir('rotated_frames')

    # if not os.path.exists('frames/dummy_file.txt'):
    #     os.mknod('frames/dummy_file.txt')

    if not os.path.exists('weights'):
        os.mkdir('weights')

    os.chdir('yolov5')

    if len(os.listdir('.')) == 0:
        print('Cloning repo... ', end='')
        git.Repo.clone_from('https://github.com/ultralytics/yolov5.git', '.')
        print('Cloned repo')

        print('Installing requirements... ', end='')
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        print('Installed requirements')

        if gpu:
            print('Installing GPU version of pytorch... ', end='')
            path = '../utils/command.txt'
            command = []
            if os.path.exists(path):
                with open(path) as source:
                    command = source.read().split()
                    command = [x for x in command if x != '' and x != ' ']

                if len(command) != 0:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torch', '-y'],
                                          stdout=subprocess.DEVNULL)
                    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torchvision', '-y'],
                                          stdout=subprocess.DEVNULL)
                    # subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torchaudio', '-y'],
                    #                       stdout=subprocess.DEVNULL)
                    print('Uninstalled torch for CPU... ', end='')

                    subprocess.check_call(command, stdout=subprocess.DEVNULL)
                    print('Installed GPU support for pytorch')
                else:
                    print('The file that should contain the command is empty, skipping')
            else:
                print('File containing command to install GPU support for pytorch not found, skipping')

    print('Got YOLOv5 Repo and installed requirements')
    os.chdir('../weights')

    if len(os.listdir('.')) == 0:
        if small and not os.path.exists('yolov5s6.pt'):
            wget.download(small_link)
        elif not small and not os.path.exists('yolov5x6.pt'):
            wget.download(xl_link)
    print('Got YOLOv5 weights')

    os.chdir('..')
    shutil.copy('utils/modified_detect.py', 'yolov5/modified_detect.py')
    os.chdir('code')


def get_video_info(filename):
    """
        Returns the frame count, the fps and the duration of the video
        :param filename: path of the file to analyze
        :return: frame_count, fps, duration and image size
    """

    video = cv2.VideoCapture(filename)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    success, frame = video.read()
    size = (frame.shape[0], frame.shape[1])
    video.release()

    return frame_count, fps, duration, size


def remove_past_runs():
    """
        Removes folder of past runs
        :return: Void
    """

    os.chdir('../yolov5')
    if os.path.exists('runs/detect'):
        os.chdir('runs/detect')
        for item in os.listdir('.'):
            shutil.rmtree(item)
        os.chdir('../..')
    os.chdir('../code')


def create_db(name):
    """
        Creates a SQLite database based on the name of the file given to the function
        :param name: name of the file for which the Database is created
        :return: The connection to the database if it was created, None otherwise
    """

    path_of_db = '../db/' + name + '.db'
    try:
        conn = sqlite3.connect(path_of_db)
    except Error as e:
        print(e)
        return None

    return conn


def create_table(conn, cmd):
    """
        Executes the sql command on the given database. Used just to create the table in the database
        :param conn: the Connection object
        :param cmd: the command to execute
        :return: True if the table is created, False otherwise
    """

    if 'CREATE TABLE' not in cmd:
        print('This is not a table creation')
        return False
    try:
        conn.cursor().execute(cmd)
    except Error as e:
        print(e)
        return False
    return True


def divide_into_frames(filename):
    """
        Runs ffmpeg to divide the video in single frames
        :param filename: name of the file to be divided
        :return: Void
    """
    print('Dividing into frames... ', end='')
    ffmpeg.input(filename).output('../frames/frame_%d_0.png', start_number=0).run(quiet=True)
    print('Divided into frames')


def rotate_frames():
    """
        Rotates all the frames by 90, 180 and 270 degrees
        :return: Void
    """
    print('Rotating frames... ', end='')
    os.chdir('../frames')
    for file in os.listdir('.'):
        if 'dummy_file' not in file:
            image = cv2.imread(file)
            cv2.imwrite('../rotated_frames/' + file[:-5] + '90.png', cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            cv2.imwrite('../rotated_frames/' + file[:-5] + '180.png', cv2.rotate(image, cv2.ROTATE_180))
            cv2.imwrite('../rotated_frames/' + file[:-5] + '270.png', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    for file in os.listdir('.'):
        if 'dummy_file' not in file:
            shutil.move(file, '../rotated_frames/' + file)
    os.chdir('../code')
    print('Rotated frames')


def delete_frames():
    """
        Deletes all the files from the frame folder
        :return: Void
    """
    os.chdir('../frames')
    for item in os.listdir('.'):
        if 'dummy_file' not in item:
            os.remove(item)
    os.chdir('../rotated_frames')
    for item in os.listdir('.'):
        if 'dummy_file' not in item:
            os.remove(item)
    os.chdir('../code')


def process_video(filename, small, gpu):
    """
        Analyzes all frames with YOLO
        :param gpu: True if the user wants to use the GPU
        :param filename: path of the file to use
        :param small: True if the user wants to use the small version of YOLO
        :return: Void
    """

    print('Processing video... ')
    weights_file = '../weights/yolov5x6.pt'
    if small:
        weights_file = '../weights/yolov5s6.pt'

    name_of_video = filename.split('/')[-1][:-4]

    remove_past_runs()
    db = create_db(name_of_video)

    if db is None:
        print('Something went wrong while creating the database')
        exit(3)

    sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS frameInfo (
                                            frameIndex integer,
                                            detections text ,
                                            coordinates text,
                                            rotation integer DEFAULT 0,
                                            tag text DEFAULT '',
                                            PRIMARY KEY(frameIndex, rotation)
                                        ); """

    table_created = create_table(db, sql_create_projects_table)
    if not table_created:
        print('Something went wrong while creating the table')
        exit(3)

    divide_into_frames(filename)
    rotate_frames()

    command = ['python', 'modified_detect.py', '--source', '../rotated_frames/', '--weights', weights_file,
               '--originalName', name_of_video, '--directory',
               '--classes', '0 14 15 16 17 18 19 20 21 22 23 77',
               '--conf-thres', '0.1'  # , '--nosave'
               ]
    if gpu:
        command.append('--device')
        command.append('0')

    os.chdir('../yolov5/')
    subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # delete_frames()
    print('Finished processing')


def transform_coordinates(result, rot, size):
    """
        Transforms the coordinates based on the rotation on the frame
        :param result: coordinates of the frame as stored in the database
        :param rot: rotation of the frame
        :param size: image size
        :return: transformed coordinates
    """

    result = result.replace(', device=\'cuda:0\')', '').replace('tensor(', '').replace('.', '') \
        .replace(',', '').replace('[', '').replace(']', '').split(' ')
    result = np.array([int(x) for x in result if x != '']).reshape((-1, 4))

    width = result[2] - result[0]
    height = result[3] - result[1]

    if rot == 180:
        for i in range(len(result)):
            tmp_array = np.array([size[0] - result[i][0] - width, size[1] - result[i][1],
                                  size[0] - result[i][2], size[1] - height + result[i][3]])
            result[i] = tmp_array
    elif rot == 90:
        for i in range(len(result)):
            point = result[i]
            tmp_array = np.array([point[1], size[1] - point[0] - width, point[3], size[1] - point[2] + width])
            result[i] = tmp_array
    elif rot == 270:
        for i in range(len(result)):
            point = result[i]
            tmp_array = np.array([size[0] - height, point[0], size[0] - point[3] + height, point[2]])
            result[i] = tmp_array

    return result


def distance(arr1, arr2):
    """
        Returns a mean between the distances between the top left corner and the bottom right corner
        :param arr1: Points of the first box
        :param arr2: Points of the second box
        :return: float
    """

    p1 = [arr1[0], arr1[1]]
    p2 = [arr2[0], arr2[1]]
    p3 = [arr1[2], arr1[3]]
    p4 = [arr2[2], arr2[3]]
    x_diff1 = abs(p1[0] - p2[0])
    x_diff2 = abs(p3[0] - p4[0])
    y_diff1 = abs(p1[1] - p2[1])
    y_diff2 = abs(p3[1] - p4[1])

    return (sqrt(x_diff1 ** 2 + y_diff1 ** 2) + sqrt(x_diff2 ** 2 + y_diff2 ** 2)) / 2


def keep_only_nearest_boxes(first_array, second_array):
    """
        Returns 2 arrays of the same length, with the corresponding boxes in the same positions
        :param first_array: first array of detections
        :param second_array: second array of detections
        :return: 2 arrays
    """

    swapped = False
    similarities = []

    if len(first_array) > len(second_array):
        swapped = True
        first_array, second_array = second_array, first_array

    for first_item in first_array:
        aux_similarity = []
        for second_item in second_array:
            aux_similarity.append(distance(first_item, second_item))
        similarities.append(aux_similarity)

    mins = []
    for item in similarities:
        aux_arr = item
        max_val = np.argmin(aux_arr)
        while aux_arr[max_val] in mins:
            aux_arr = np.delete(aux_arr, max_val)
            max_val = np.argmin(aux_arr)
        mins.append(aux_arr[max_val])

    indexes = []
    for i in range(len(mins)):
        indexes.append(similarities[i].index(mins[i]))
    second_return = [second_array[val] for val in indexes]

    if swapped:
        return second_return, first_array

    return first_array, second_return


def interpolate_coordinates(filename, indexes, skipping_indexes, results):
    """
        Interpolates the coordinates of bounding boxes on frames in which objects were probably not detected due to low
            confidence in the prediction
        :param indexes: indexes of the frames in which something was found
        :param skipping_indexes: indexes of the frames that skipped
        :return: Void
    """

    filename = filename.split('/')[-1][:-4]
    difference_threshold = 100
    for i in range(len(skipping_indexes)):
        idx = skipping_indexes[i]
        prev_idx = skipping_indexes[i] - 1
        diff = indexes[idx] - indexes[prev_idx]
        sorted_couples_1, sorted_couples_2 = keep_only_nearest_boxes(results[idx][2],
                                                                     results[prev_idx][2])
        coordinates = []

        for k in range(diff - 1):
            aux = []
            for j in range(len(sorted_couples_1)):
                top_left_x_difference = sorted_couples_2[j][0] - sorted_couples_1[j][0]
                top_left_y_difference = sorted_couples_2[j][1] - sorted_couples_1[j][1]
                bottom_right_x_difference = sorted_couples_2[j][2] - sorted_couples_1[j][2]
                bottom_right_y_difference = sorted_couples_2[j][3] - sorted_couples_1[j][3]
                if top_left_x_difference > difference_threshold or top_left_y_difference > difference_threshold \
                        or bottom_right_x_difference > difference_threshold \
                        or bottom_right_y_difference > difference_threshold:
                    continue
                top_x_step = top_left_x_difference / diff
                top_y_step = top_left_y_difference / diff
                bottom_x_step = bottom_right_x_difference / diff
                bottom_y_step = bottom_right_y_difference / diff
                aux.append([int(sorted_couples_1[j][0] + top_x_step * (k + 1)),
                            int(sorted_couples_1[j][1] + top_y_step * (k + 1)),
                            int(sorted_couples_1[j][2] + bottom_x_step * (k + 1)),
                            int(sorted_couples_1[j][3] + bottom_y_step * (k + 1))])
            coordinates.append(aux)

        for w in range(diff - 1):
            index = indexes[idx] - diff + 1 + w
            sql = """UPDATE frameInfo 
                     SET frameIndex = ?, detections = ?, coordinates = ?, rotation = ?, tag = ?
                     WHERE frameIndex = ? AND rotation = ?"""
            data = (index, 'person, ' * len(coordinates[w]), str(coordinates[w]), 0, '', index, 0)
            path = '../db/' + filename + '.db'
            conn = sqlite3.connect(path)
            if conn is None:
                print('Connection to database error')
                exit(3)
            conn.cursor().execute(sql, data)
            conn.commit()


def post_process(filename, size):
    """

        :param filename: path of the file
        :param size: image size
        :return: Void
    """

    frame_interval = 10
    name_of_video = filename.split('/')[-1][:-4]
    conn = sqlite3.connect('../db/' + name_of_video + '.db')

    # Transform coordinates to get real coordinates
    sql = """SELECT * FROM frameInfo WHERE detections <> '' ORDER BY frameIndex"""
    if conn is None:
        print('Error in connection to DB')
        exit(2)
    results = conn.cursor().execute(sql).fetchall()
    results = [list(item) for item in results]
    new_results = []
    indexes = [item[0] for item in results]

    print('Transforming coordinates of rotated frames... ', end='')
    for result in results:
        if result[3] != 0:
            result[2] = transform_coordinates(result[2], result[3], size)
        else:
            result[2] = result[2].replace(', device=\'cuda:0\')', '').replace('tensor(', '').replace('.', '') \
                .replace(',', '').replace('[', '').replace(']', '').split(' ')
            result[2] = np.array([int(x) for x in result[2] if x != '']).reshape((-1, 4))

        new_results.append(result)

    print('Transformed all coordinates')
    # Using the informations to understand if there are no detections for less then a few frames and infer the position
    # of the eventual missed bounding box

    print('Interpolating bounding boxes of probably skipped frames')
    skipping_indexes = []
    for i in range(1, len(indexes)):
        if 1 < indexes[i] - indexes[i - 1] < frame_interval:
            skipping_indexes.append(i)
    interpolate_coordinates(filename, indexes, skipping_indexes, new_results)


def find_background():
    return 0


def move_to_trash(filename):
    """
        Moves to trash the DB based on the filename given
        :param filename: path of the processed video
        :return: Void
    """
    name_of_video = filename.split('/')[-1][:-4]
    name_of_db = '../db/' + name_of_video + '.db'
    send2trash(name_of_db)
