import os
import mimetypes
import wget
import git
import subprocess
import sys
import shutil


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

    name = filename.split('/')
    name = name[-1][:-4]
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
        git.Repo.clone_from('https://github.com/ultralytics/yolov5.git')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                              stdout=subprocess.DEVNULL)

    os.chdir('..')
    if not os.path.exists('weights'):
        os.mkdir('weights')

    os.chdir('weights')

    if len(os.listdir('.')) == 0:
        if small and not os.path.exists('yolov5s6.pt'):
            wget.download(small_link)
        elif not os.path.exists('yolov5x6.pt'):
            wget.download(xl_link)

    os.chdir('..')
    shutil.copy('utils/modified_detect.py', 'yolov5/modified_detect.py')
    os.chdir('code')



