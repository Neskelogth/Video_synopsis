# standard library imports
import os
import mimetypes
import subprocess
import sys
import shutil
import sqlite3
from sqlite3 import Error
# import time

# external modules imports
import wget
import git
import cv2
from send2trash import send2trash
# import concurrent.futures
import ffmpeg
import numpy as np


def check_requirements(file_path):
    """
        Checks if the file to examine exists, if it's a video and if it is an mp4 file
        :param file_path: path of the file to check
        :return: True if the conditions are met, False otherwise
    """

    detect_path = '../utils/modified_detect.py'

    if not os.path.exists(file_path):
        print('The file does not exist')
        return False

    if not mimetypes.guess_type(file_path)[0].startswith('video'):
        print('The file is not a video')
        return False

    if file_path[-3:] != 'mp4':
        print('The file is not an .mp4 video')
        return False

    if not os.path.exists(detect_path):
        print('Missing important files, something went wrong')
        return False

    return True


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

    # if not os.path.exists('frames/dummy_file.txt'):
    #     os.mknod('frames/dummy_file.txt')

    if not os.path.exists('weights'):
        os.mkdir('weights')

    os.chdir('yolov5')

    if len(os.listdir('.')) == 0:
        git.Repo.clone_from('https://github.com/ultralytics/yolov5.git', '.')
        print('Cloned repo')
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        print('Installed requirements')
        if gpu:
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torch', '-y'], stdout=subprocess.DEVNULL)
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torchvision', '-y'],
                                  stdout=subprocess.DEVNULL)
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'torchaudio', '-y'],
                                  stdout=subprocess.DEVNULL)
            print('Uninstalled torch')
            path = '../utils/command.txt'
            with open(path) as source:
                command = source.read().split()
                subprocess.check_call(command, stdout=subprocess.DEVNULL)
            print('GPU support')

    print('Got YOLOv5 Repo and installed requirements')
    os.chdir('../weights')

    if len(os.listdir('.')) == 0:
        if small and not os.path.exists('yolov5s6.pt'):
            wget.download(small_link)
        elif not os.path.exists('yolov5x6.pt'):
            wget.download(xl_link)
    print('Got YOLOv5 weights')

    os.chdir('..')
    shutil.copy('utils/modified_detect.py', 'yolov5/modified_detect.py')
    os.chdir('code')


def get_video_info(filename):
    """
        Returns the frame count, the fps and the duration of the video
        :param filename: path of the file to analyze
        :return: frame_count, fps, duration
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
    print('Deleted already present files')


def create_db(name):
    """
        Creates a SQLite database based on the name of the file given to the function
        :param name: name of the file for which the Database is created
        :return: The connection to the database if it was created, None otherwise
    """

    path_of_db = '../db/' + name + '.db'
    try:
        conn = sqlite3.connect(path_of_db)
        # print('Connection established')
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
        c = conn.cursor()
        c.execute(cmd)
    except Error as e:
        print(e)
        return False
    print('Table created')
    return True


def divide_into_frames(filename):
    """
        Runs ffmpeg to divide the video in single frames
        :param filename: name of the file to be divided
        :return: Void
    """
    os.chdir('../frames')
    for item in os.listdir('.'):
        if 'dummy_file' not in item:
            os.remove(item)

    ffmpeg.input(filename).output('../frames/frame_%d_0.png', start_number=0).run(quiet=True)
    os.chdir('../code/')
    print('Divided into frames')


def rotate_frames():
    os.chdir('../frames')
    for file in os.listdir('.'):
        if 'dummy_file' not in file:
            image = cv2.imread(file)
            cv2.imwrite(file[:-5] + '90.png', cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            cv2.imwrite(file[:-5] + '180.png', cv2.rotate(image, cv2.ROTATE_180))
            cv2.imwrite(file[:-5] + '270.png', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    os.chdir('../code')
    print('Rotated frames')


def process_video(filename, small):
    """
        Analyzes all frames with YOLO
        :param filename: path of the file to use
        :param small: True if the user wants to use the small version of YOLO
        :return: Void
    """

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

    # divide_into_frames(filename)
    # rotate_frames()

    command = ['python', 'modified_detect.py', '--source', '../frames/', '--weights', weights_file,
               '--originalName', name_of_video, '--directory', '--device', '0',
               '--classes', '0 14 15 16 17 18 19 20 21 22 23 77', '--nosave']

    os.chdir('../yolov5/')
    subprocess.check_call(command, stdout=subprocess.DEVNULL)
    print('Finished processing')


def post_process(path, size):
    """"""
    pass


def move_to_trash(filename):
    """
        Moves to trash the DB based on the filename given
        :param filename: path of the processed video
        :return: Void
    """
    name_of_video = filename.split('/')[-1][:-4]
    name_of_db = '../db/' + name_of_video + '.db'
    send2trash(name_of_db)
