import os
import mimetypes


def check_requirements(file_path):

    """
    Checks if the file to examine exists, if it's a video and if it is an mp4 file
    :param file_path: path of the file to check
    :return: True if the conditions are met, False otherwise
    """

    if not os.path.exists(file_path):
        print('The file does not exist')
        return False

    if not mimetypes.guess_type(file_path)[0].startswith('video'):
        print('The file is not a video')
        return False

    if file_path[-3:] != 'mp4':
        print('The file is not an .mp4 video')
        return False

    return True



