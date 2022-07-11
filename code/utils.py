# standard library imports
import os
import mimetypes
import subprocess
import sys
import time
import shutil
import sqlite3
from sqlite3 import Error

# external modules imports
import wget
import git
import cv2
from send2trash import send2trash
# import concurrent.futures
import ffmpeg

# external file imports
# from YOLO_integration import analyze_frame


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

    name = filename.split('/')[-1][:-4]
    if os.path.exists('../db/' + name):
        return True
    return False


def get_necessary_files(small):
    """
        Gets all the necessary files for the script to work
        :param small: True if the small model of YOLO has to be used
        :return: Void
    """

    small_link = 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt'
    xl_link = 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt'

    os.chdir('..')
    if not os.path.exists('yolov5'):
        os.mkdir('yolov5')

    os.chdir('yolov5')

    if len(os.listdir('.')) == 0:
        git.Repo.clone_from('https://github.com/ultralytics/yolov5.git', '.')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                              stdout=subprocess.DEVNULL)

    os.chdir('..')
    print('Got YOLOv5 Repo')
    if not os.path.exists('weights'):
        os.mkdir('weights')

    os.chdir('weights')

    if len(os.listdir('.')) == 0:
        if small and not os.path.exists('yolov5s6.pt'):
            wget.download(small_link)
        elif not os.path.exists('yolov5x6.pt'):
            wget.download(xl_link)

    os.chdir('..')
    print('Got YOLOv5 weights')
    shutil.copy('utils/modified_detect.py', 'yolov5/modified_detect.py')
    os.chdir('code')
    print('Moved modified detect file')


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


def move_detect():
    """
        Moves the file modified_detect file to the yolov5 folder for later
        :return: Void
    """
    if not os.path.exists('../yolov5/modified_detect.py'):
        os.chdir('..')
        os.rename('utils/modified_detect.py', 'yolov5/modified_detect.py')
        os.chdir('code')


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
        print('Connection established')
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
    os.chdir('../frames_of_video/')
    for item in os.listdir('.'):
        if 'dummy_file' not in item:
            os.remove(item)
    ffmpeg.input(filename).output('../frames_of_video/frame_%d.png', start_number=0).run(quiet=True)
    os.chdir('../code/')
    print('Divided into frames')


def process_video(filename, small):

    MAX_NUMBER_OF_PROCESSES = 3
    WEIGHTS_FILE = '../weights/yolov5x6.pt'
    if small:
        WEIGHTS_FILE = '../weights/yolov5s6.pt'

    name_of_video = filename.split('/')[-1][:-4]
    print(name_of_video)

    remove_past_runs()
    db = create_db(name_of_video)

    if db is None:
        print('Something went wrong while creating the database')
        exit(3)

    sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS frameInfo (
                                            frameIndex integer PRIMARY KEY,
                                            detections text ,
                                            coordinates text
                                        ); """

    table_created = create_table(db, sql_create_projects_table)
    if not table_created:
        print('Something went wrong while creating the table')
        exit(3)

    # divide_into_frames(filename)  # for test purposes, it's commented

    # found_something = []
    current_processes = []
    frames = [item for item in os.listdir('../frames_of_video/')]
    frames = frames[1:]  # remove dummy_file from the list
    i = len(frames) - 1
    os.chdir('../yolov5/')
    # Processing frames in parallel mode
    while len(frames) > 0:

        if i % 100 == 0:
            print(i, 'frames remaining')

        for process in current_processes:
            if process.poll() is not None:
                current_processes.remove(process)
            if len(current_processes) >= MAX_NUMBER_OF_PROCESSES:
                time.sleep(1)  # wait for process to end

        frameIndex = frames[i].replace('frame_', '').replace('.png', '')
        frame_path = '../frames_of_video/' + frames[i]

        p = subprocess.Popen(['python', 'modified_detect.py', '--source', frame_path, '--weights', WEIGHTS_FILE,
                              '--idx', frameIndex, '--originalName', name_of_video, '--line-thickness', '1',
                              '--hide-conf', '--hide-labels'
                              # , '--nosave'
                              ],
                             stderr=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL
                             )
        current_processes.append(p)
        frames.pop()
        i -= 1

    print('Finished processing')

    if len(rows) == 0:
        print('Finished')
    else:
        for row in rows:
            print(row)


def save_copy_for_test(fps, size):

    db = sqlite3.connect('../db/cut.db')
    if db is None:
        return

    sql = """SELECT * FROM frameInfo"""
    frame_info = db.cursor().execute(sql)
    rows = frame_info.fetchall()

    print(len(rows))

    os.chdir('../frames_of_video')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv2.VideoWriter('test.mp4', fourcc, fps, size)
    i = 0
    for item in os.listdir():
        if 'dummy_file' in item:
            continue
        sql = """SELECT * FROM frameInfo WHERE detections IS NOT NULL"""
        frame_info = db.cursor().execute(sql)
        rows = frame_info.fetchall()


def move_to_trash(filename):
    """
        Moves to trash the DB based on the filename given
        :param filename: path of the processed video
        :return: Void
    """
    name_of_video = filename.split('/')[-1][:-4]
    name_of_db = '../db/' + name_of_video + '.db'
    send2trash(name_of_db)
