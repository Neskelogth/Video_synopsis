import os
from utils import check_requirements, get_necessary_files, check_if_db_exists


# initial declarations of constants
data_path = '../data/08/2018-12-03 08-15-00~08-19-59.mp4'
# input_small = input('Type Y if you want to use the small version of YOLO, N if you want to use the xl version: ')
small = False

# if input_small == 'Y':
#     small = True


# if the requirements are not satisfied, exit with code 2
if not check_requirements(data_path):
    exit(2)


# if the video has already been processed, don't process it again
already_processed_file = check_if_db_exists(data_path)
# Get all necessary files to run the project
get_necessary_files(small)

# if the video has not been processed go through entire pipeline
if not already_processed_file:
    pass

