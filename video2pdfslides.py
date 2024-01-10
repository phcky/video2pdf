import os
import time
import cv2
import imutils
import shutil
import img2pdf
import glob
import pathlib
from PIL import Image

############# Define constants

OUTPUT_SLIDES_DIR = f"./output"

FRAME_RATE = 1  # no.of frames per second that needs to be processed, fewer the count faster the speed
WARMUP = FRAME_RATE  # initial number of frames to be skipped
FGBG_HISTORY = FRAME_RATE * 15  # no.of frames in background object
VAR_THRESHOLD = 16  # Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
DETECT_SHADOWS = False  # If true, the algorithm will detect shadows and mark them.
MIN_PERCENT = 0.1  # min % of diff between foreground and background to detect if motion has stopped
MAX_PERCENT = 10  # max % of diff between foreground and background to detect if frame is still in motion


def get_frames(video_path):
    '''A fucntion to return the frames from a video located at video_path
    this function skips frames as defined in FRAME_RATE'''

    # open a pointer to the video file initialize the width and height of the frame
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise Exception(f'unable to open file {video_path}')

    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_time = 0
    frame_count = 0
    print("total_frames: ", total_frames)
    print("FRAME_RATE", FRAME_RATE)

    # loop over the frames of the video
    while True:
        # grab a frame from the video

        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)  # move frame to a timestamp
        frame_time += 1 / FRAME_RATE

        (_, frame) = vs.read()
        # if the frame is None, then we have reached the end of the video file
        if frame is None:
            break

        frame_count += 1
        yield frame_count, frame_time, frame

    vs.release()


def detect_unique_screenshots(video_path, output_folder_screenshot_path):
    ''''''
    # Initialize fgbg a Background object with Parameters
    # history = The number of frames history that effects the background subtractor
    # varThreshold = Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model. This parameter does not affect the background update.
    # detectShadows = If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false.

    fgbg = cv2.createBackgroundSubtractorMOG2(history=FGBG_HISTORY, varThreshold=VAR_THRESHOLD,
                                              detectShadows=DETECT_SHADOWS)

    captured = False
    start_time = time.time()
    (W, H) = (None, None)

    screenshoots_count = 0
    for frame_count, frame_time, frame in get_frames(video_path):
        orig = frame.copy()  # clone the original frame (so we can save it later),
        frame = imutils.resize(frame, width=600)  # resize the frame
        mask = fgbg.apply(frame)  # apply the background subtractor

        # apply a series of erosions and dilations to eliminate noise
        #            eroded_mask = cv2.erode(mask, None, iterations=2)
        #            mask = cv2.dilate(mask, None, iterations=2)

        # if the width and height are empty, grab the spatial dimensions
        if W is None or H is None:
            (H, W) = mask.shape[:2]

        # compute the percentage of the mask that is "foreground"
        p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100

        # if p_diff less than N% then motion has stopped, thus capture the frame

        if p_diff < MIN_PERCENT and not captured and frame_count > WARMUP:
            captured = True
            filename = f"{screenshoots_count:03}_{round(frame_time / 60, 2)}.png"

            path = os.path.join(output_folder_screenshot_path, filename)
            print("saving {}".format(path))
            cv2.imwrite(path, orig)
            screenshoots_count += 1

        # otherwise, either the scene is changing or we're still in warmup
        # mode so let's wait until the scene has settled or we're finished
        # building the background model
        elif captured and p_diff >= MAX_PERCENT:
            captured = False
    print(f'{screenshoots_count} screenshots Captured!')
    print(f'Time taken {time.time() - start_time}s')
    return


def initialize_output_folder(video_path):
    '''Clean the output folder if already exists'''
    output_folder_screenshot_path = f"{OUTPUT_SLIDES_DIR}/{video_path.rsplit('/')[-1].split('.')[0]}"

    if os.path.exists(output_folder_screenshot_path):
        shutil.rmtree(output_folder_screenshot_path)

    os.makedirs(output_folder_screenshot_path, exist_ok=True)
    print('initialized output folder', output_folder_screenshot_path)
    return output_folder_screenshot_path


def convert_screenshots_to_pdf(output_folder_screenshot_path):
    output_pdf_path = f"{OUTPUT_SLIDES_DIR}/{video_path.rsplit('/')[-1].split('.')[0]}" + '.pdf'
    print('output_folder_screenshot_path', output_folder_screenshot_path)
    print('output_pdf_path', output_pdf_path)
    print('converting images to pdf..')
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(sorted(glob.glob(f"{output_folder_screenshot_path}/*.jpg"))))
    print('Pdf Created!')
    print('pdf saved at', output_pdf_path)


def get_size_format(b, factor=1024, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"


def compress_img(image_name, new_size_ratio=0.8, quality=95, width=None, height=None, to_jpg=True):
    img = Image.open(image_name)
    print(" [*] Image shape:", img.size)
    image_size = os.path.getsize(image_name)
    print(" [*] Size before compression:", get_size_format(image_size))
    if new_size_ratio < 1.0:
        img = img.resize((int(img.size[0] * new_size_ratio), int(img.size[1] * new_size_ratio)), Image.LANCZOS)
        print(" [+] New Image shape:", img.size)
    elif width and height:
        img = img.resize((width, height), Image.LANCZOS)
        print(" [+] New Image shape:", img.size)
    filename, ext = os.path.splitext(image_name)
    if to_jpg:
        new_filename = f"{filename}_compressed.jpg"
    else:
        new_filename = f"{filename}_compressed{ext}"
    try:
        img.save(new_filename, quality=quality, optimize=True)
    except OSError:
        img = img.convert("RGB")
        img.save(new_filename, quality=quality, optimize=True)
    print(" [+] New file saved:", new_filename)
    new_image_size = os.path.getsize(new_filename)
    print(" [+] Size after compression:", get_size_format(new_image_size))
    saving_diff = new_image_size - image_size
    print(f" [+] Image size change: {saving_diff / image_size * 100:.2f}% of the original image size.")


def calculate_image_similarity(image1_path, image2_path):
    # Read images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find key points and descriptors with SIFT
    key_points1, descriptors1 = sift.detectAndCompute(image1, None)
    key_points2, descriptors2 = sift.detectAndCompute(image2, None)

    # FLANN parameters for feature matching
    flann_index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)

    # Use FLANN to perform feature matching
    flann = cv2.FlannBasedMatcher(flann_index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test for good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Calculate similarity based on the number of good matches
    similarity = len(good_matches) / len(key_points1)

    return similarity


def remove_duplicate_images(image_paths):
    prev_image = None
    for idx, img_path in enumerate(image_paths):
        if idx > 0:
            similarity = calculate_image_similarity(prev_image, img_path)
            # print (prev_image, "and", img_path, "similarity is:", similarity)
            if similarity > 0.45:
                os.remove(img_path)
                print(f"Removed duplicate image: {img_path}")
                continue
            compress_img(prev_image)
            os.remove(prev_image)
        prev_image = img_path
    compress_img(image_paths[-1])
    os.remove(image_paths[-1])


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir_name = './input'  # The folder in which the video to convert is located
    for file_name in os.listdir(input_dir_name):
        # Check if the file is an .mp4 file
        if file_name.lower().endswith('.mp4'):
            print(f"正在转换：{file_name}")
            video_path = str(pathlib.Path(input_dir_name, file_name))
            output_folder_screenshot_path = initialize_output_folder(video_path)
            detect_unique_screenshots(video_path, output_folder_screenshot_path)
            image_paths = glob.glob(f"{output_folder_screenshot_path}/*.png")
            image_paths = [x.replace('\\', '/') for x in image_paths]
            remove_duplicate_images(image_paths)
            # Convert extracted images to PDF
            convert_screenshots_to_pdf(output_folder_screenshot_path)
