from utils import check_files, get_necessary_files, check_if_db_exists, process_video, get_video_info
from utils import post_process, move_to_trash, find_background
from utils import keep_only_nearest_boxes
import time
import numpy as np


start = time.time()

# initial declarations of constants
data_path = '../data/08/cut_more.mp4'
# data_path = '../data/08/2018-12-03 08-15-00~08-19-59.avi'
# input_small = input('Type y if you want to use the small version of YOLO, n if you want the xl version: ').lower()
# input_gpu = input('Type y if you want to use the GPU, n otherwise: ').lower()
small = True  # test purposes
gpu = True
# input_delete = input('Do you want to delete the db file after the process?(Y/N) ').lower()
delete_db = False

# if input_small == 'y':
#     small = True

# if input_gpu == 'n':
#     gpu = False

# if input_delete == 'y':
#     delete_db = True


# if the requirements are not satisfied, exit with code 2
file_path = check_files(data_path)
if file_path is None:
    exit(2)


print('Checked all requirements')

# if the video has already been processed, don't process it again
already_processed_file = check_if_db_exists(data_path)
# print('File processed = ', already_processed_file)
# Get all necessary files to run the project
get_necessary_files(small, gpu)
print('All files are set')

frame_count, fps, duration, size = get_video_info(data_path)
print(f'frame_count = {frame_count}, fps = {fps}, duration = {duration}, size = {size}')

# if the video has not been processed go through entire pipeline
if not already_processed_file:
    process_video(data_path, small, gpu)
    # pass  # test purposes

post_process(data_path, size)
bg = find_background()
if delete_db:
    move_to_trash(data_path)

print(f'Finished in {(time.time() - start) / 60} minutes')
