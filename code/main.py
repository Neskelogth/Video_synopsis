from utils import check_files, get_necessary_files, check_if_db_exists, get_video_info, process_video, post_process
from utils import find_background, move_to_trash, remove_all_frames
import time


start = time.time()

# initial declarations of constants
data_path = '../data/08/cut_more_2.mp4'
# input_small = input('Type y if you want to use the small version of YOLO, n if you want the xl version: ').lower()
# input_gpu = input('Type y if you want to use the GPU, n otherwise: ').lower()
small = True  # test purposes
gpu = True
# input_delete = input('Do you want to delete the db file after the process?(Y/N) ').lower()
delete_db = False
# input_delete_frames = input('Do you want to delete the frames used while processing when the process finishes?
# (Y/N) ').lower()
delete_frames = False

# if input_small == 'y':
#     small = True

# if input_gpu == 'n':
#     gpu = False

# if input_delete == 'y':
#     delete_db = True

# if input_delete_frames == 'y':
#     delete_frames = True


# if the requirements are not satisfied, exit with code 2
file_path = check_files(data_path)
if file_path is None:
    exit(2)


print('Checked all requirements')

# if the video has already been processed, don't process it again
already_processed_file = check_if_db_exists(file_path)
# print('File processed = ', already_processed_file)
# Get all necessary files to run the project
get_necessary_files(small, gpu)
print('All files are set')

frame_count, fps, duration, size = get_video_info(file_path)
print(f'frame_count = {frame_count}, fps = {fps}, duration = {duration}, size = {size}')

# if the video has not been processed go through entire pipeline
if not already_processed_file:
    process_video(file_path, small, gpu)
    post_process(file_path, size)
    find_background(file_path)

if delete_db:
    move_to_trash(file_path)

if delete_frames:
    remove_all_frames()

print(f'Finished in {(time.time() - start) / 60} minutes')

