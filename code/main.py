import time
from utils import check_files, get_necessary_files, check_if_db_exists, get_video_info, process_video, post_process
from utils import find_background, associate_tag, save_video, move_to_trash, remove_all_frames, refine_tags


if __name__ == '__main__':
    start = time.time()

    # initial declarations of constants
    data_path = ''
    gpu = True
    delete_db = False
    delete_frames = False
    output_process = True
    input_gpu = input('Type y if you want to use the GPU, n otherwise: ').lower()
    input_delete = input('Do you want to delete the db file after the process?(Y/N) ').lower()
    input_delete_frames = \
        input('Do you want to delete the frames used while processing when the process finishes? (Y/N) ').lower()
    input_output = input('Do you want to see the output of YOLO on the console?(Y/N) ').lower()
    data_path = input('Input the name of the file to be examined ')
    data_path = '../data/' + data_path

    if input_output == 'y':
        output_process = True

    if input_delete == 'y':
        delete_db = True

    if input_delete_frames == 'y':
        delete_frames = True

    if input_gpu == 'n':
        gpu = False
    else:
        input("Please remember to insert the command to install pytorch with CUDA support on the file called command in the utils folder. Press enter to continue...")

    # if the requirements are not satisfied, exit with code 2
    file_path = check_files(data_path)
    if file_path is None:
        exit(2)

    print('Checked all requirements')

    # if the video has already been processed, don't process it again
    already_processed_file = check_if_db_exists(file_path)

    # Get all necessary files to run the project
    get_necessary_files(gpu)
    print('All files are set')

    fps, size, frame_count = get_video_info(file_path)
    print(f'fps = {fps}, size = {size}, frame_count = {frame_count}')

    # if the video has not been processed go through entire pipeline
    if not already_processed_file:
        process_video(file_path, gpu, output_process)
        post_process(file_path, size)
        find_background(file_path)
        associate_tag(file_path)
        refine_tags(file_path)

    save_video(file_path, fps, size)

    if delete_db:
        move_to_trash(file_path)

    if delete_frames:
        remove_all_frames()

    time = time.time() - start
    print(f'Finished in {time} seconds')
