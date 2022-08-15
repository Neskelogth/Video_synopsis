# standard library imports
import os
import subprocess
import sys
import shutil
import sqlite3
from sqlite3 import Error
from collections import Counter
from math import sqrt, floor

# external modules imports
import wget
import git
import cv2
from send2trash import send2trash
import ffmpeg
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import magic


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

    mime = magic.Magic(mime=True)
    mimetype = mime.from_file(file_path)
    if not mimetype or mimetype.find('video') == -1:
        print('The file is not a video')
        return None

    if file_path[-3:] != 'mp4':
        print('The file is not in mp4 format, please input an mp4 file')
        return None

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


def get_necessary_files(gpu):
    """
        Gets all the necessary files for the script to work
        :param gpu: True if the user wants to use the GPU for YOLO
        :return: Void
    """

    link = 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt'

    os.chdir('..')
    if not os.path.exists('yolov5'):
        os.mkdir('yolov5')

    if not os.path.exists('frames'):
        os.mkdir('frames')

    if not os.path.exists('rotated_frames'):
        os.mkdir('rotated_frames')

    if not os.path.exists('weights'):
        os.mkdir('weights')
    
    if not os.path.exists('output'):
        os.mkdir('output')
        os.mkdir('output/final_frames')
    else:
        if not os.path.exists('output/final_frames'):
            os.mkdir('output/final_frames')

    open('output/ffmpeg_frames.txt', 'w').close()

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
            if os.path.exists(path):
                with open(path) as source:
                    command = source.read().split()
                    command = [x for x in command if x != '' and x != ' ']

                if len(command) != 0:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torch', '-y'],
                                          stdout=subprocess.DEVNULL)
                    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torchvision', '-y'],
                                          stdout=subprocess.DEVNULL)
                    print('Uninstalled torch for CPU... ', end='')

                    subprocess.check_call(command, stdout=subprocess.DEVNULL)
                    print('Installed GPU support for pytorch')
                else:
                    print('The file that should contain the command is empty, skipping')
            else:
                print('File containing command to install GPU support for pytorch not found, skipping')

    os.chdir('../weights')

    if len(os.listdir('.')) == 0:
        if not os.path.exists('yolov5s6.pt'):
            wget.download(link)

    os.chdir('..')
    shutil.copy('utils/modified_detect.py', 'yolov5/modified_detect.py')
    os.chdir('code')


def get_video_info(filename):
    """
        Returns the fps and the size of the video
        :param filename: path of the file to analyze
        :return: fps and image size
    """

    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    success, frame = video.read()
    size = (frame.shape[0], frame.shape[1])
    video.release()

    return fps, size, frame_count


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
    os.chdir('../output/final_frames')
    for file in os.listdir('.'):
        if 'dummy_file' not in file:
            os.remove(file)
    os.chdir('../../code')


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
    print('Rotating frames... ')
    os.chdir('../frames')
    list_of_files = os.listdir('.')
    for i in tqdm(range(len(list_of_files) - 1)):
        if 'dummy_file' not in list_of_files[i]:
            image = cv2.imread(list_of_files[i])
            cv2.imwrite('../rotated_frames/' + list_of_files[i][:-5] + '90.png',
                        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            cv2.imwrite('../rotated_frames/' + list_of_files[i][:-5] + '180.png',
                        cv2.rotate(image, cv2.ROTATE_180))
            cv2.imwrite('../rotated_frames/' + list_of_files[i][:-5] + '270.png',
                        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

    for i in range(len(list_of_files)):
        if 'dummy_file' not in list_of_files[i]:
            shutil.move(list_of_files[i], '../rotated_frames/' + list_of_files[i])
    os.chdir('../code')
    print('Rotated frames')


def process_video(filename, gpu, out):
    """
        Analyzes all frames with YOLO
        :param out:
        :param gpu: True if the user wants to use the GPU
        :param filename: path of the file to use
        :return: Void
    """

    print('Deleting frames of past runs...', end='')
    remove_all_frames()
    print('Deleted')

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
               '--classes', '0', '--conf-thres', '0.1'  # , '--nosave'
               ]
    if gpu:
        command.append('--device')
        command.append('0')

    print('Processing video. This operation may take some time, please wait... ', end='')
    os.chdir('../yolov5/')
    if out:
        subprocess.check_call(command)
    else:
        subprocess.check_call(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    print('Finished processing')













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
            .replace(',', '').replace('[', '').replace(']', '').replace('\n', '').replace('array(', '') \
            .replace(')', '').split(' ')
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


def keep_biggest_boxes(coords, prev_coords, index):
    """

        :param coords:
        :param prev_coords:
        :param index:
        :return:
    """

    coords_areas = []
    prev_coords_areas = []

    for i in range(len(coords)):
        coords_areas.append((coords[i][3] - coords[i][1]) * (coords[i][2] - coords[i][0]))
        prev_coords_areas.append((prev_coords[i][3] - prev_coords[i][1]) * (prev_coords[i][2] - prev_coords[i][0]))

    diffs = [(coords_areas[i] - prev_coords_areas[i]) > 0 for i in range(len(coords_areas))]
    c = Counter(diffs)
    if c[True] > c[False]:
        return index - 1

    return index


def filter_out_repeated_indexes(results):
    """
        Filters the results removing detections where YOLO found something in the original and the rotated frames
        :param results: results of the queried database
        :return: Filtered results
    """

    to_delete = []

    for i in range(len(results) - 1, 1, -1):
        if results[3] != 0:
            if results[i - 1][0] == results[i][0] and len(results[i - 1][2]) >= len(results[i][2]):
                to_delete.append(i)
            elif results[i - 1][0] == results[i][0] and len(results[i][2]) >= len(results[i - 1][2]):
                to_delete.append(i - 1)

    for item in to_delete:
        results.pop(item)

    return results


def handle_outliers(results, frame_interval):
    """

        :param frame_interval:
        :param results:
        :return:
    """

    coordinates = [item[2] for item in results]
    indexes = [item[0] for item in results]
    diff_threshold = 100
    to_delete = []

    prev_coords = coordinates[0]
    for i in range(1, len(results)):
        all_dists = []
        for j in range(len(prev_coords)):
            dists = []
            for k in range(len(coordinates[i])):
                dists.append(distance(prev_coords[j], coordinates[i][k]))
            all_dists.append(dists)

        outlier = False
        truth = []
        for item in all_dists:
            if all(inner_item > diff_threshold for inner_item in item) \
                    and (indexes[i] - indexes[i - 1]) < frame_interval:
                truth.append(True)
            else:
                truth.append(False)

        if all(truth):
            outlier = True
            to_delete.append(results[i][0])

        if not outlier:
            prev_coords = coordinates[i]

    new_results = []
    for i in range(len(results)):
        if results[i][0] not in to_delete:
            new_results.append(results[i])

    return new_results


def keep_only_nearest_boxes(first_array, second_array):
    """
        Returns 2 arrays of the same length, with the corresponding boxes in the same positions
        :param first_array: first array of detections
        :param second_array: second array of detections
        :return: 2 arrays
    """

    distances = []
    first_array, second_array = second_array, first_array

    for first_item in first_array:
        aux_sim = []
        for second_item in second_array:
            aux_sim.append(distance(first_item, second_item))
        distances.append(aux_sim)

    indexes_to_keep = []
    for item in distances:
        aux_val = item.copy()
        min_idx = np.argmin(aux_val)
        while item.index(aux_val[min_idx]) in indexes_to_keep:
            aux_val.pop(min_idx)
            min_idx = np.argmin(aux_val)
        indexes_to_keep.append(min_idx)

    second_array = [second_array[i] for i in indexes_to_keep]

    return second_array


def discard_probably_non_interesting_boxes(results, idx, frame_interval):
    """

        :param results:
        :param idx:
        :param frame_interval:
        :return:
    """

    for i in range(frame_interval):

        sentinel1 = False
        sentinel2 = False

        if idx + i < len(results):
            sentinel1 = True
            current_result = results[idx + i][2]
        if idx - 1 > 0:
            sentinel2 = True
            prev_result = results[idx - 1 + i][2]

        if sentinel2 and sentinel1 and len(prev_result) < len(current_result):
            results[idx + i][2] = keep_only_nearest_boxes(current_result, prev_result)
    return results[idx]


def filter_boxes(results):
    """

        :param results:
        :return:
    """
    ####################################################################################################################
    frame_interval = 33
    ####################################################################################################################

    for i in range(len(results)):
        if len(results[i][2]) == len(results[i - 1][2]):
            continue

        next_coordinates_lengths = []
        prev_coordinates_lengths = []
        for j in range(frame_interval):
            if i + j < len(results):
                next_coordinates_lengths.append(len(results[i + j][2]))
            if i - j > 0:
                prev_coordinates_lengths.append(len(results[i - j][2]))

        c = Counter(next_coordinates_lengths)
        c2 = Counter(prev_coordinates_lengths)

        if (c[len(results[i - 1][2])] > c[len(results[i][2])] or c2[len(results[i - 1][2])] > c2[len(results[i][2])]) \
                and ((c[len(results[i][2])] + c[len(results[i - 1][2])] == frame_interval) and
                     (c2[len(results[i][2])] + c2[len(results[i - 1][2])] == frame_interval)):
            results[i] = discard_probably_non_interesting_boxes(results, i, frame_interval)

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


def interpolate_coordinates(skipping_indexes, results):
    """
        Interpolates the coordinates of bounding boxes on frames in which objects were probably not detected due to low
            confidence in the prediction
        :param results: results of the processing step
        :param filename: name of the video being processed
        :param skipping_indexes: indexes of the frames that skipped
        :return: Void
    """

    # Interpolation of coordinates
    for i in range(len(skipping_indexes)):
        item = skipping_indexes[i]
        diff = results[item][0] - results[item - 1][0]
        idx = results[item - 1][0]
        if len(results[item][1]) < len(results[item - 1][1]):
            detections = results[item][1]
        else:
            detections = results[item][1]
        coordinates = results[item][2]
        prev_coordinates = results[item - 1][2]
        first_array, second_array = filter_and_sort(prev_coordinates, coordinates)

        for k in range(diff - 1):
            aux_result = []
            for j in range(len(first_array)):
                top_x_step = (second_array[j][0] - first_array[j][0]) / diff
                top_y_step = (second_array[j][1] - first_array[j][1]) / diff
                bot_x_step = (second_array[j][2] - first_array[j][2]) / diff
                bot_y_step = (second_array[j][3] - first_array[j][3]) / diff
                aux_coords = [round(first_array[j][0] + top_x_step * (k + 1)),
                              round(first_array[j][1] + top_y_step * (k + 1)),
                              round(first_array[j][2] + bot_x_step * (k + 1)),
                              round(first_array[j][3] + bot_y_step * (k + 1))]
                aux_result.append(np.array(aux_coords))
            results.append([idx + k + 1, detections, aux_result, 0, ''])

    return results


def save_coordinates(results, filename):

    """

        :param filename:
        :param results:
        :return: Void
    """
    sql = """UPDATE frameInfo
                 SET frameIndex = ?, detections = ?, coordinates = ?, rotation = 0, tag = ?
                 WHERE frameIndex = ? AND rotation = 0"""

    conn = sqlite3.connect('../db/' + filename + '.db')
    if conn is None:
        print('Error in connecting to the database')
        exit(3)
    for item in results:
        data = (item[0], item[1], str(item[2]), '', item[0])
        conn.cursor().execute(sql, data)
    conn.commit()


def post_process(filename, size):
    """

        :param filename: path of the file
        :param size: image size
        :return: Void
    """

    frame_interval = 30
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

    print('Filtering out results...', end='')
    results = filter_out_repeated_indexes(results)
    print('Done')

    print('Transforming coordinates and more filtering...', end='')
    results = transform_coordinates(results, size)
    print('Done')

    print('Handling outliers...', end='')
    results = handle_outliers(results, frame_interval)
    print('Handled outliers')

    print('Filtering boxes... ', end='')
    results = filter_boxes(results)
    print('Done')

    indexes = [item[0] for item in results]
    skipping_indexes = []

    print('Interpolating coordinates...', end='')
    for i in range(1, len(indexes)):
        if 1 < indexes[i] - indexes[i - 1] < frame_interval:
            skipping_indexes.append(i)

    results = interpolate_coordinates(skipping_indexes, results)
    print('Interpolated')

    save_coordinates(results, name_of_video)

    sql = """DELETE FROM frameInfo WHERE rotation <> 0"""
    conn.cursor().execute(sql)
    conn.commit()


def find_background(filename):
    """

        :return: Void
    """

    sql = """SELECT frameIndex 
             FROM frameInfo 
             WHERE detections = ''
             ORDER BY frameIndex"""

    name_of_video = filename.split('/')[-1][:-4]
    conn = sqlite3.connect('../db/' + name_of_video + '.db')
    if conn is None:
        print('Error in database connection')
        exit(3)
    print('Searching for background frame...')
    results = conn.cursor().execute(sql).fetchall()
    indexes = [item[0] for item in results]

    names = ['frame_' + str(idx) + '_0.png' for idx in indexes]
    os.chdir('../rotated_frames')
    ssims = []
    prev_image = cv2.imread(names[0])
    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

    for i in tqdm(range(1, len(names))):
        image = cv2.imread(names[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (score, diff) = ssim(prev_image, image, full=True)
        prev_image = image
        ssims.append(score)

    max_sim = np.argmax(ssims)
    name = names[max_sim + 1]
    shutil.copy(name, '../utils/' + name_of_video + '_background.png')

    os.chdir('../code')
    print('Finished searching')


def get_new_tags(coords, max_tag, dead_tags, last_coords, iddx):

    """

        :param coords:
        :param max_tag:
        :param dead_tags:
        :param last_coords:
        :return:
    """

    all_dists = []
    min_dists = []
    possible_tags = []

    for i in range(max_tag + 1):
        aux_string = 'tag_' + str(i)
        if not dead_tags[aux_string]:
            possible_tags.append(aux_string)

    prev_coords = []
    for item in possible_tags:
        prev_coords.append(last_coords[item])

    for i in range(len(prev_coords)):
        dists = []
        for j in range(len(coords)):
            dists.append(distance(prev_coords[i], coords[j]))
        all_dists.append(dists)
        min_dists.append(min(dists))

    all_dists = np.array(all_dists)
    max_dist = max(min_dists)
    sorted_mins = [''] * len(min_dists)
    new_tag_list = [''] * len(coords)
    counter = 0
    thresh = 100

    while any(item < max_dist + 1 for item in min_dists) and counter < 100000:
        min_idx = np.argmin(min_dists)
        counter += 1
        if min_dists[min_idx] < thresh:
            couple_min_idx = np.argmin(all_dists[min_idx])
            string = str(min_idx) + ' ' + str(couple_min_idx)
            min_dists[min_idx] = max_dist + 1
            sorted_mins[min_idx] = string
        else:
            break

    if counter == 100000:
        print('max_count')

    counter = 0
    done = 0

    for i in range(len(sorted_mins)):
        if sorted_mins[i] != '':
            idx = sorted_mins[i].split(' ')
            old_idx, new_idx = int(idx[0]), int(idx[1])
            new_tag_list[new_idx] = possible_tags[old_idx]
            done += 1
        if done == len(coords):
            break

    for i in range(len(new_tag_list)):
        if new_tag_list[i] == '':
            new_tag_list[i] = 'tag_' + str(max_tag + counter + 1) + ' '
            counter += 1

    counter = 0
    if len(coords) > len(new_tag_list):
        for i in range(len(coords) - len(new_tag_list)):
            string = 'tag_' + str(max_tag + counter + 1)
            counter += 1
            new_tag_list.append(string)

    num_tags = [int(item.replace('tag_', '')) for item in new_tag_list]
    max_tag = max(num_tags)

    return new_tag_list, max_tag


def associate_tag(filename):
    """

        :param filename:
        :return:
    """

    print('Associating tags...', end='')
    name_of_video = filename.split('/')[-1][:-4]
    conn = sqlite3.connect('../db/' + name_of_video + '.db')
    if conn is None:
        print('Error in connecting to database')
        exit(3)

    sql = """SELECT * FROM frameInfo WHERE detections <> '' ORDER BY frameIndex"""
    results = conn.cursor().execute(sql).fetchall()
    results = [list(item) for item in results]
    indexes = [item[0] for item in results]

    coordinates = [item[2].replace('tensor', '').replace('array', '').replace(')', '').replace('(', '').replace(',', '')
                       .replace('[', '').replace(']', '').replace('.', '').replace('device=\'cuda:0\'', '').split(' ')
                   for item in results]
    for i in range(len(coordinates)):
        coordinates[i] = [int(item) for item in coordinates[i] if item != '']
        coordinates[i] = np.array(coordinates[i]).reshape(-1, 4)

    tags_used = []
    tags_per_frame = []

    string = ''
    number_first_coordinates = len(coordinates[0])
    dead_tags = {}
    last_found = {}
    last_time_coordinates = {}
    for i in range(number_first_coordinates):
        aux_string = 'tag_' + str(i)
        string += aux_string + ' '
        tags_used.append(aux_string)
        dead_tags[aux_string] = False
        last_found[aux_string] = indexes[0]
        last_time_coordinates[aux_string] = coordinates[0][i]

    max_tag = int(tags_used[-1].split('_')[-1])
    tags_per_frame.append(string)
    interval_to_die = 30

    for i in range(1, len(indexes)):
        if indexes[i] - indexes[i - 1] == 1:

            current_tags, last_tag = get_new_tags(coordinates[i], max_tag, dead_tags, last_time_coordinates, i)
            for j in range(len(current_tags)):
                if current_tags[j] not in tags_used:
                    dead_tags[current_tags[j].replace(' ', '')] = False
                last_found[current_tags[j].replace(' ', '')] = indexes[i]
                last_time_coordinates[current_tags[j].replace(' ', '')] = coordinates[i][j]
            for j in range(len(tags_used)):
                if not dead_tags[tags_used[j]] \
                        and tags_used[j] not in current_tags \
                        and indexes[i] - last_found[tags_used[j]] > interval_to_die:
                    dead_tags[tags_used[j]] = True
            string = ' '.join(current_tags)
            tags_per_frame.append(string)
            max_tag = max(max_tag, last_tag)
        else:
            number_of_tags = len(coordinates[i])
            aux_string = ''
            dead_tags = dict.fromkeys(dead_tags, True)
            for j in range(number_of_tags):
                s = 'tag_' + str(max_tag + j + 1)
                aux_string += s + ' '
                dead_tags[s] = False
                last_found[s] = indexes[i]
                last_time_coordinates[s] = coordinates[i][j]

            tags_per_frame.append(aux_string)
            max_tag += number_of_tags

    sql = """UPDATE frameInfo SET tag = ? WHERE frameIndex = ?"""

    for i in range(len(indexes)):
        conn.cursor().execute(sql, (str(tags_per_frame[i]), indexes[i]))

    conn.commit()
    print('Finished')


def find_segment(tag, tags_per_segment):

    """

        :param tag:
        :param tags_per_segment:
        :return:
    """

    for key in tags_per_segment:
        if tag in tags_per_segment[key]:
            return key

    return -1


def counter_in_segment(counters, segment, tags_per_segment):

    """

        :param counters:
        :param segment:
        :param tags_per_segment:
        :return:
    """

    needed_tags = tags_per_segment[segment]
    current_counters = {}

    for item in needed_tags:
        current_counters[item] = counters[item]

    return current_counters


def pick_most_probable_tag(candidates, counters, current_tags, results, idx):

    """

        :param candidates:
        :param counters:
        :param current_tags:
        :return:
    """

    max_counters = {}

    for item in candidates:
        if item not in max_counters.keys():
            max_counters[item] = 0

    frame_interval = 7

    for i in range(frame_interval):
        if idx - i > 0:
            examined_tags = results[idx - i][4].split(' ')
            examined_tags = [item for item in examined_tags if item != '' and item != ' ' and item in candidates]
            for item in examined_tags:
                max_counters[item] += 1

    ordered_tags = dict(sorted(max_counters.items(), key=lambda item: item[1]))
    final_tag = list(ordered_tags.keys())[0]

    return final_tag


def substitute_tag(tag, current_tags, counters_in_segment, segment, tags_per_segment, results, i):

    """

        :param tag:
        :param current_tags:
        :param counters_in_segment:
        :param segment:
        :param tags_per_segment:
        :param results:
        :param i:
        :return:
    """

    frame_min = 30
    candidate_tags = []
    tags_per_segment = tags_per_segment[segment]

    for key in tags_per_segment:
        if counters_in_segment[key] > frame_min:
            candidate_tags.append(key)

    current_tags.remove(tag)

    if len(candidate_tags) == 1:
        current_tags.append(candidate_tags[0])

    if len(candidate_tags) > 1:
        current_tags.append(pick_most_probable_tag(candidate_tags, counters_in_segment, current_tags, results, i))

    return current_tags


def handle_index(tag, results, tags_per_segment, counters):
    """

        :param counters:
        :param tag:
        :param results:
        :param tags_per_segment:
        :return:
    """
    ####################################################################################################################
    frame_interval = 21
    ####################################################################################################################

    for i in range(len(results)):
        current_tags = results[i][4].split(' ')
        current_tags = [item for item in current_tags if item != '']
        if tag in current_tags:
            delete_condition = False

            coordinates = results[i][2].replace('array', '').replace('(', '').replace(')', '').replace(',', '') \
                .replace('.', '').replace('[', '').replace(']', '').split(' ')
            coordinates = np.array([int(item) for item in coordinates if item != '']).reshape(-1, 4)
            index_in_tags = current_tags.index(tag)
            current_len = len(coordinates)
            for j in range(frame_interval + 1):
                # next_lengths = [current_len]
                prev_lengths = [current_len]
                # if i + j + 1 < len(results):
                #     next_coordinates = results[i + j + 1][2].replace('array', '').replace('(', '').replace(')', '')\
                #         .replace(',', '').replace('.', '').replace('[', '').replace(']', '').split(' ')
                #     next_coordinates = np.array([int(item) for item in next_coordinates if item != '']).reshape(-1, 4)
                #     next_lengths.append(len(next_coordinates))

                if i - j - 1 > 0:
                    prev_coordinates = results[i - j - 1][2].replace('array', '').replace('(', '').replace(')', '')\
                        .replace(',', '').replace('.', '').replace('[', '').replace(']', '').split(' ')
                    prev_coordinates = np.array([int(item) for item in prev_coordinates if item != '']).reshape(-1, 4)
                    prev_lengths.append(len(prev_coordinates))

                # c1 = Counter(next_lengths)
                c2 = Counter(prev_lengths)

                # c1[current_len] < floor(frame_interval / 2) \
                # and

                if c2[current_len] < floor(frame_interval / 2) and len(coordinates) > 1:
                    delete_condition = True

            if delete_condition:
                current_tags.pop(index_in_tags)
                coordinates = np.delete(coordinates, index_in_tags, axis=0)
            else:
                segment = find_segment(tag, tags_per_segment)
                counters_in_segment = counter_in_segment(counters, segment, tags_per_segment)
                prev_current = current_tags
                current_tags = substitute_tag(tag, current_tags, counters_in_segment, segment,
                                              tags_per_segment, results, i)
                if len(prev_current) != len(current_tags):
                    idx = prev_current.index(tag)
                    coordinates = np.delete(coordinates, idx, axis=0)
            joined_tags = ' '.join(current_tags)
            results[i][2] = str(coordinates)
            results[i][4] = joined_tags

    return results


def refine_tags(filename):
    """

        :param filename:
        :return:
    """

    print('Refining tags...', end='')
    name_of_video = filename.split('/')[-1][:-4]

    sql = """SELECT * FROM frameInfo WHERE coordinates <> '' ORDER BY frameIndex"""

    conn = sqlite3.connect('../db/' + name_of_video + '.db')
    if conn is None:
        print('Error in connecting to the database')
        exit(3)

    thresh = 30
    results = conn.cursor().execute(sql).fetchall()
    results = [list(item) for item in results]

    tags = [item[4] for item in results]
    indexes = [item[0] for item in results]
    counters = {}

    for item in tags:
        current_tags = item.split(' ')
        current_tags = [tag for tag in current_tags if tag != '']
        for tag in current_tags:
            if tag not in counters.keys():
                counters[tag] = 1
            else:
                counters[tag] += 1
    
    aux_tags = tags[0].split(' ')
    tags_per_segment = {
        0: [item for item in aux_tags if item != '']
    }
    segment_counter = 0
    
    for i in range(1, len(results)):
        current_tags = tags[i].split(' ')
        current_tags = [tag for tag in current_tags if tag != '']
        if indexes[i] - indexes[i - 1] == 1:
            for tag in current_tags: 
                if tag not in tags_per_segment[segment_counter]:
                    tags_per_segment[segment_counter].append(tag)
        else:
            segment_counter += 1
            tags_per_segment[segment_counter] = current_tags

    for key in counters.keys():
        if counters[key] < thresh:
            results = handle_index(key, results, tags_per_segment, counters)

            

    for i in range(1, len(results)):
        pass

    sql = """UPDATE frameInfo SET tag = ?, coordinates = ? WHERE frameIndex = ?"""
    for result in results:
        conn.cursor().execute(sql, (result[4], str(result[2]), result[0]))

    conn.commit()
    print('Done')


def proceed(counters, fpt):
    """

        :param counters:
        :param fpt:
        :return:
    """
    lengths = []
    counter = 0
    for key in fpt:
        lengths.append(counters[counter] == len(fpt[key]))
        counter += 1

    if all(lengths):
        return False

    return True


def overlap(box1, box2):

    """

        :param box1:
        :param box2:
        :return:
    """

    overlap_on_x = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    overlap_on_y = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)

    return overlap_on_y * overlap_on_x


def compute_overlap(box, array, tolerance=0.1):

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if len(array) > 0:
        for item in array:
            item_area = (item[2] - item[0]) * (item[3] - item[1])
            min_area = min(box_area, item_area)
            over = overlap(box, item) / min_area
            if over > tolerance:
                return True

    return False


def get_next_box(results, tag, indexes):

    box = []
    index_to_find = indexes[0]

    for item in results:
        if item[0] == index_to_find:
            current_tags = item[4].split(' ')
            current_tags = [tg.replace(' ', '') for tg in current_tags if tg != '' and tg != ' ']
            index_in_coords = current_tags.index(tag)
            coords = item[2].replace('[', '').replace(']', '').replace('array', '').replace('(', '').replace(')', '')\
                .replace(',', '').replace('\n', '').split(' ')
            coords = [it for it in coords if it != '']
            coords = np.array([int(it) for it in coords]).reshape(-1, 4)
            box = coords[index_in_coords]
            break

    return box


def different(box1, box2):
    """

        :param box1:
        :param box2:
        :return:
    """
    if len(box1) != len(box2):
        return True

    for i in range(len(box1)):
        if box1[i] != box2[i]:
            return True

    return False


def build_image(present_boxes, bg_copy, dict, alpha=0.7):
    """

        :param present_boxes:
        :param bg_copy:
        :param dict:
        :param alpha:
        :return:
    """

    os.chdir('../rotated_frames/')
    frame_indexes = []

    for key in dict:
        frame_indexes.append(key)

    counter = 0
    for box in present_boxes:
        over = []
        img = 'frame_' + str(frame_indexes[counter]) + '_0.png'
        counter += 1
        for inner_box in present_boxes:
            if different(box, inner_box):
                over.append(compute_overlap(box, [inner_box]))

        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        subimage = image[box[1]:box[3], box[0]:box[2]]

        if any(over):
            sub_bg = bg_copy[box[1]:box[3], box[0]:box[2]]
            final_subimage = cv2.addWeighted(sub_bg, 1 - alpha, subimage, alpha, 0)
            bg_copy[box[1]:box[3], box[0]:box[2]] = final_subimage
        else:
            bg_copy[box[1]:box[3], box[0]:box[2]] = subimage

    os.chdir('../output')

    return bg_copy


def save_video(filename, fps, size):
    """

        :param filename:
        :param fps:
        :param size:
        :return:
    """
    print('Creating video output...', end='')
    name_of_video = filename.split('/')[-1][:-4]
    sql = """SELECT DISTINCT tag
             FROM frameInfo
             WHERE detections <> ''
             ORDER BY frameIndex"""

    conn = sqlite3.connect('../db/' + name_of_video + '.db')
    if conn is None:
        print('Error while connecting to the database')
        exit(3)

    results = conn.cursor().execute(sql).fetchall()
    results = [item for item in results]
    tags = [item[0].replace('[', '').replace(']', '').replace('\'', '').replace(',', '').split(' ') for item in results
            if item != '']
    list_of_tags = []
    duration = str(1/fps)

    for i in range(len(tags)):
        tags[i] = [item for item in tags[i] if item != '']
        for item in tags[i]:
            list_of_tags.append(item)

    tag_set_manual = []

    for item in list_of_tags:
        if item not in tag_set_manual:
            tag_set_manual.append(item)

    frames_per_tag = {}
    start_index_per_tag = {}

    sql = """SELECT *
             FROM frameInfo
             WHERE tag <> ''
             ORDER BY frameIndex"""

    results = conn.cursor().execute(sql).fetchall()
    results = [list(item) for item in results]

    for tag in tag_set_manual:
        frames_per_tag[tag] = []
        aux_res = results
        lowest = results[len(results) - 1][0]
        for item in aux_res:
            if tag in item[4] and item[0] < lowest:
                lowest = item[0]
        start_index_per_tag[tag] = lowest

    for item in results:
        current_tags = item[4].split(' ')
        current_tags = [item.replace(' ', '') for item in current_tags if item != '']
        index = item[0]
        for idx in current_tags:
            frames_per_tag[idx].append(index)

    number_of_tags = len(tag_set_manual)
    os.chdir('../utils')
    background = 0

    for file in os.listdir('.'):
        if name_of_video in file:
            background = cv2.imread(file)
            background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)

    if type(background) is not np.ndarray:
        print('Something went wrong while reading the background')
        exit(4)

    os.chdir('../output')

    counters = [0] * number_of_tags
    counter = 0
    new_frames = {}

    while proceed(counters, frames_per_tag):
        present_boxes = {}
        for key in frames_per_tag:
            if len(frames_per_tag[key]) > 0:

                box = get_next_box(results, key, frames_per_tag[key])
                idx = frames_per_tag[key].pop(0)
                present_boxes[idx] = box

        new_frames[counter] = present_boxes
        counter += 1

    for key in new_frames:
        bg_copy = background.copy()
        frame_boxes = new_frames[key]
        current_boxes = []
        for frame_index in frame_boxes:
            current_boxes.append(frame_boxes[frame_index])

        new_frame = build_image(current_boxes, bg_copy, new_frames[key])
        cv2.imwrite('final_frames/frame_' + str(key) + '.png', new_frame)
        with open('ffmpeg_frames.txt', 'a') as destination:
            destination.write('file \'final_frames/frame_' + str(key) + '.png\'\n')
            destination.write('duration ' + duration + '\n')

    out_name = name_of_video + '_synopsis.mp4'
    command = ['ffmpeg', '-f', 'concat', '-i', 'ffmpeg_frames.txt', '-framerate', str(int(fps)),
               '-c:v', 'copy', out_name, '-y']

    subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.chdir('../code')
    print('Finished')


def move_to_trash(filename):
    """
        Moves to trash the DB based on the filename given
        :param filename: path of the processed video
        :return: Void
    """
    name_of_video = filename.split('/')[-1][:-4]
    name_of_db = '../db/' + name_of_video + '.db'
    send2trash(name_of_db)