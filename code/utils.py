# standard library imports
import os
import mimetypes
import subprocess
import sys
import shutil
import sqlite3
from sqlite3 import Error
from collections import Counter
from math import sqrt, ceil

# external modules imports
import wget
import git
import cv2
from send2trash import send2trash
# import concurrent.futures
import ffmpeg
import numpy as np
from skimage.metrics import structural_similarity


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


def remove_all_frames():
    """
        Deletes all the frames from the folders used in previous runs
        :return: Void
    """
    os.chdir('../frames')
    for file in os.listdir('.'):
        if 'dummy_file' not in file:
            os.remove(file)
    os.chdir('../rotated_frames')
    for file in os.listdir('.'):
        if 'dummy_file' not in file:
            os.remove(file)
    os.chdir('../utils')
    for file in os.listdir('.'):
        if '.png' in file:
            os.remove(file)
    os.chdir('../code')


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


def process_video(filename, small, gpu):
    """
        Analyzes all frames with YOLO
        :param gpu: True if the user wants to use the GPU
        :param filename: path of the file to use
        :param small: True if the user wants to use the small version of YOLO
        :return: Void
    """

    print('Deleting frames of past runs...', end='')
    # remove_all_frames()
    print('Deleted')

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

    print('Processing video... ', end='')
    os.chdir('../yolov5/')
    subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Finished processing')


def filter_out_results(results):
    """
        Filters the results removing detections where YOLO found something in the original and the rotated frames
        :param results: results of the queried database
        :return: Filtered results
    """

    to_delete = []

    for i in range(len(results) - 1, 1, -1):
        if results[3] != 0:
            if results[i - 1][0] == results[i][0]:
                to_delete.append(i)

    for item in to_delete:
        results.pop(item)

    return results


def transform_coordinates(results, size):
    """

        :param results:
        :param size:
        :return:
    """
    new_results = []
    height = size[0]
    width = size[1]

    for result in results:
        current_rotation = result[3]
        coordinates = result[2].replace('tensor(', '').replace(', device=\'cuda:0\')', '').replace('.', '') \
            .replace(',', '').replace('[', '').replace(']', '').replace('\n', '').split(' ')
        coordinates = np.array([int(x) for x in coordinates if x != '']).reshape(-1, 4)
        result[2] = coordinates
        if current_rotation == 0:
            new_results.append(result)
            continue
        transformed_coordinates = []

        if current_rotation == 180:
            for coordinate in coordinates:
                aux_array = [width - coordinate[2], height - coordinate[3],
                             width - coordinate[0], height - coordinate[1]]
                transformed_coordinates.append(aux_array)
        if current_rotation == 90:
            for coordinate in coordinates:
                aux_array = [width - coordinate[3], coordinate[0],
                             width - coordinate[1], coordinate[2]]
                transformed_coordinates.append(aux_array)
        if current_rotation == 270:
            for coordinate in coordinates:
                aux_array = [coordinate[1], height - coordinate[2],
                             coordinate[3], height - coordinate[0]]
                transformed_coordinates.append(aux_array)

        aux_result = [result[0], result[1], np.array(transformed_coordinates), current_rotation, '']
        new_results.append(aux_result)

    return new_results


def distance(arr1, arr2):
    """
        Returns a mean between the distances between the top left corner and the bottom right corner
        :param arr1: Points of the first box
        :param arr2: Points of the second box
        :return: float
    """
    top_x_diff = arr2[0] - arr1[0]
    top_y_diff = arr2[1] - arr1[1]
    bot_x_diff = arr2[2] - arr1[2]
    bot_y_diff = arr2[3] - arr1[3]

    return (sqrt(top_x_diff ** 2 + top_y_diff ** 2) + sqrt(bot_x_diff ** 2 + bot_y_diff ** 2)) / 2


def keep_only_nearest_boxes(first_array, second_array):
    """
        Returns 2 arrays of the same length, with the corresponding boxes in the same positions
        :param first_array: first array of detections
        :param second_array: second array of detections
        :return: 2 arrays
    """

    distances = []

    if len(first_array) > len(second_array):
        first_array, second_array = second_array, first_array

    for first_item in first_array:
        aux_sim = []
        for second_item in second_array:
            aux_sim.append(distance(first_item, second_item))
        distances.append(aux_sim)

    indexes_to_keep = []
    for item in distances:
        aux_val = item
        min_idx = np.argmin(aux_val)
        while min_idx in indexes_to_keep:
            aux_val.pop(min_idx)
            min_idx = np.argmin(aux_val)
        indexes_to_keep.append(min_idx)

    second_array = [second_array[i] for i in indexes_to_keep]

    return second_array


def discard_probably_non_interesting_boxes(results, idx, farme_interval):
    """

        :param results:
        :param idx:
        :param farme_interval:
        :return:
    """

    for i in range(farme_interval):
        current_result = results[idx + i][2]
        prev_result = results[idx - 1 + i][2]
        if len(prev_result) < len(current_result):
            results[idx + i][2] = keep_only_nearest_boxes(current_result, prev_result)
    return results[idx]


def filter_boxes(results):
    """

        :param results:
        :return:
    """

    frame_interval = 7

    for i in range(len(results)):
        if len(results[i][2]) == len(results[i - 1][2]):
            continue

        next_coordinates_lengths = []
        for j in range(frame_interval):
            if i + j < len(results):
                next_coordinates_lengths.append(len(results[i + j][2]))

        c = Counter(next_coordinates_lengths)
        if c[len(results[i - 1][2])] > c[len(results[i][2])] \
                and (c[len(results[i][2])] + c[len(results[i - 1][2])] > frame_interval / 2):
            results[i] = discard_probably_non_interesting_boxes(results, i, frame_interval)

    # for item in results:
    #     print(item)

    return results


def filter_and_sort(prev_coord, coord):
    """

        :param prev_coord:
        :param coord:
        :return:
    """

    swapped = False
    distances = []

    if len(prev_coord) <= len(coord):
        first_array = np.array(prev_coord)
        second_array = np.array(coord)
    else:
        swapped = True
        first_array = np.array(coord)
        second_array = np.array(prev_coord)

    for item in first_array:
        aux_sim = []
        for second_item in second_array:
            aux_sim.append(distance(item, second_item))
        distances.append(aux_sim)

    keeping_indexes = []

    for item in distances:
        min_idx = np.argmin(item)
        while min_idx in keeping_indexes:
            item.pop(min_idx)
            min_idx = np.argmin(item)
        keeping_indexes.append(min_idx)

    second_array = [second_array[i] for i in keeping_indexes]

    if swapped:
        return second_array, first_array

    return first_array, second_array


def interpolate_and_save_coordinates(filename, skipping_indexes, results):
    """
        Interpolates the coordinates of bounding boxes on frames in which objects were probably not detected due to low
            confidence in the prediction
        :param results: results of the processing step
        :param filename: name of the video being processed
        :param skipping_indexes: indexes of the frames that skipped
        :return: Void
    """
    # Interpolation of coordinates
    for item in skipping_indexes:
        diff = results[item][0] - results[item - 1][0]
        idx = results[item - 1][0]
        if len(results[item][1]) < len(results[item - 1][1]):
            detections = results[item][1]
        else:
            detections = results[item][1]
        coordinates = results[item][2]
        prev_coordinates = results[item - 1][2]
        first_array, second_array = filter_and_sort(prev_coordinates, coordinates)

        for i in range(diff - 1):
            aux_result = []
            for j in range(len(first_array)):
                top_x_step = (second_array[j][0] - first_array[j][0]) / diff
                top_y_step = (second_array[j][1] - first_array[j][1]) / diff
                bot_x_step = (second_array[j][2] - first_array[j][2]) / diff
                bot_y_step = (second_array[j][3] - first_array[j][3]) / diff
                aux_coords = [ceil(first_array[j][0] + top_x_step * (i + 1)),
                              ceil(first_array[j][1] + top_y_step * (i + 1)),
                              ceil(first_array[j][2] + bot_x_step * (i + 1)),
                              ceil(first_array[j][3] + bot_y_step * (i + 1))]
                aux_result.append(np.array(aux_coords))
            results.append([idx + i + 1, detections, aux_result, 0, ''])

    # Saving results
    sql = """UPDATE frameInfo
             SET frameIndex = ?, detections = ?, coordinates = ?, rotation = ?, tag = ?
             WHERE frameIndex = ? AND rotation = ?"""
    conn = sqlite3.connect('../db/' + filename + '.db')
    if conn is None:
        print('Error in connecting to the database')
        exit(3)
    for item in results:
        data = (item[0], item[1], str(item[2]), 0, '', item[0], 0)
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

    sql = """SELECT * 
             FROM frameInfo 
             WHERE detections <> '' 
             ORDER BY frameIndex"""

    if conn is None:
        print('Error in connection to DB')
        exit(2)

    # Remove results where YOLO found something in more than one possible rotation
    results = conn.cursor().execute(sql).fetchall()
    results = [list(item) for item in results]
    results = filter_out_results(results)
    # transforming coordinates of rotated frames
    results = transform_coordinates(results, size)
    results = filter_boxes(results)
    indexes = [item[0] for item in results]
    skipping_indexes = []

    for i in range(1, len(indexes)):
        if 1 < indexes[i] - indexes[i - 1] < frame_interval:
            skipping_indexes.append(i)

    interpolate_and_save_coordinates(name_of_video, skipping_indexes, results)
    for item in results:
        print(item)


def find_background(filename):
    """

        :return: Void
    """

    sql = """SELECT frameIndex 
             FROM frameInfo 
             WHERE detections = '' AND rotation = 0
             ORDER BY frameIndex"""

    name_of_video = filename.split('/')[-1][:-4]
    conn = sqlite3.connect('../db/' + name_of_video + '.db')
    if conn is None:
        print('Error in database connection')
        exit(3)
    print('Searching for background frame...', end='')
    results = conn.cursor().execute(sql).fetchall()
    indexes = [item[0] for item in results]

    names = ['frame_' + str(idx) + '_0.png' for idx in indexes]
    os.chdir('../rotated_frames')
    ssims = []

    for i in range(1, len(names)):
        image_1 = cv2.imread(names[i - 1])
        image_2 = cv2.imread(names[i])
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(image_1, image_2, full=True)
        ssims.append(score)

    max_sim = np.argmax(ssims)
    name = names[max_sim + 1]
    shutil.copy(name, '../utils/' + name)

    os.chdir('../code')
    print('Finished searching')


def move_to_trash(filename):
    """
        Moves to trash the DB based on the filename given
        :param filename: path of the processed video
        :return: Void
    """
    name_of_video = filename.split('/')[-1][:-4]
    name_of_db = '../db/' + name_of_video + '.db'
    send2trash(name_of_db)
