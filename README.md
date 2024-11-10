
## **Katna**: Tool for automating video keyframe extraction, video compression, Image Autocrop and Smart image resize tasks

<img src="docs/source/logo.png" alt="logo" width="200"/>

### Resources 
* Homepage and Reference: <https://katna.readthedocs.io/>

# Video Segmentation by Keyframe Similarity

This README provides a guide on how to segment a video into smaller clips based on the similarity of keyframes extracted from the video. The process involves extracting keyframes, calculating similarity between frames, and splitting the video accordingly.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9 (or later)
- FFmpeg
- Git
- Git LFS

## Installation

1. **Activate the Python environment:**
   ```bash
   conda activate py39
   ```

2. **Install necessary packages:**
   ```bash
   sudo apt-get update && sudo apt-get install ffmpeg cbm git-lfs
   pip install katna moviepy opencv-python
   ```

3. **Clone the Katna repository (optional):**
   ```bash
   git clone https://github.com/svjack/Katna.git
   ```

## Usage

### 1. Download a Sample Video

Download a sample video from a source like Bilibili using a tool like `BBDown`:

```bash
.\BBDown.exe https://www.bilibili.com/video/BV14UD6YFEh1
mv '.\【原神·尘歌壶】翠黛峰一体化_扫冬峰 峰上人间 _ 摹本分享.mp4' BV14UD6YFEh1.mp4
```

### 2. Extract Keyframes and Segment the Video

Use the provided Python script to extract keyframes and segment the video based on similarity:

```python
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import re
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import hashlib
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter

def ensure_folder_exists(folder_path):
    """
    确保文件夹存在，如果存在则删除并重新创建。

    :param folder_path: 文件夹路径
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def extract_all_frames(video_file_path, output_folder):
    """
    从视频文件中提取所有帧，并将它们保存到指定路径，按照帧的时间戳进行命名。

    :param video_file_path: 视频文件的路径
    :param output_folder: 保存帧的输出文件夹路径
    """
    ensure_folder_exists(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc="提取所有帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # 按照帧的时间戳命名文件
        frame_name = f"{frame_count:06d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()

def extract_video_keyframes(video_file_path, no_of_frames=12):
    """
    从视频文件中提取关键帧，并将它们保存到临时文件夹中。

    :param video_file_path: 视频文件的路径
    :param no_of_frames: 要提取的关键帧数量，默认为 12
    :return: 保存关键帧的临时文件夹路径
    """
    # 创建临时文件夹
    temp_folder = f"temp_keyframes_{hashlib.md5(video_file_path.encode()).hexdigest()}"
    ensure_folder_exists(temp_folder)

    # 初始化视频模块
    vd = Video()

    # 初始化磁盘写入器以保存数据到指定位置
    diskwriter = KeyFrameDiskWriter(location=temp_folder)

    # 提取关键帧
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames,
        file_path=video_file_path,
        writer=diskwriter
    )

    return temp_folder

def calculate_similarity(image1, image2):
    """
    计算两张图像的相似度。

    :param image1: 第一张图像
    :param image2: 第二张图像
    :return: 相似度值
    """
    # 计算像素点的绝对差的平均值
    similarity = np.mean(np.abs(image1.astype(float) - image2.astype(float)))
    return -1 * similarity

def find_most_similar_frame(keyframe_path, all_frames_folder, step=10):
    """
    找到与给定关键帧最相似的帧。

    :param keyframe_path: 关键帧的路径
    :param all_frames_folder: 所有帧的文件夹路径
    :param step: 每隔多少帧比较一次，默认为 10
    :return: 最相似帧的文件名
    """
    # 读取关键帧
    keyframe = cv2.imread(keyframe_path)
    if keyframe is None:
        raise ValueError("无法读取关键帧")

    # 初始化最大相似度和对应的文件名
    max_similarity = -10000000000
    most_similar_frame_name = None

    # 遍历所有帧，每隔 step 帧比较一次
    frame_names = sorted(os.listdir(all_frames_folder))
    for frame_name in tqdm(frame_names[::step], desc="寻找最相似帧"):
        frame_path = os.path.join(all_frames_folder, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # 计算相似度
        similarity = calculate_similarity(keyframe, frame)

        # 更新最大相似度和对应的文件名
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_frame_name = frame_name

    return most_similar_frame_name

def longest_increasing_subsequence(arr):
    """
    动态规划求解最长递增子序列
    """
    if not arr:
        return []

    n = len(arr)
    dp = [1] * n
    for i in tqdm(range(1, n), desc="计算最长递增子序列"):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    max_len = max(dp)
    result = []
    for i in tqdm(range(n - 1, -1, -1), desc="构建最长递增子序列"):
        if dp[i] == max_len:
            result.append(arr[i])
            max_len -= 1

    result.reverse()
    return result

def find_longest_increasing_subsequence(file_list):
    """
    找到最长递增子序列
    """
    # 提取数值后缀
    suffixes = [(extract_numeric_suffix(file[0]), extract_numeric_suffix(file[1]), file) for file in file_list]

    # 按照第一个数值后缀排序
    suffixes.sort(key=lambda x: x[0])

    # 提取第二个数值后缀
    second_suffixes = [suffix[1] for suffix in suffixes]

    # 找到最长递增子序列
    longest_subseq = longest_increasing_subsequence(second_suffixes)

    # 找到对应的文件名
    result = []
    for suffix in tqdm(suffixes, desc="构建最长递增子序列结果"):
        if suffix[1] in longest_subseq:
            result.append(suffix[2])
            longest_subseq.remove(suffix[1])

    return result

def create_interval_videos(video_file_path, result, output_folder):
    """
    根据 find_longest_increasing_subsequence 返回的列表，将视频分割成若干个间隔视频，并输出到指定文件夹。

    :param video_file_path: 视频文件的路径
    :param result: find_longest_increasing_subsequence 返回的列表
    :param output_folder: 输出间隔视频的文件夹路径
    """
    ensure_folder_exists(output_folder)

    # 提取数值后缀
    suffixes = [(extract_numeric_suffix(file[0]), extract_numeric_suffix(file[1])) for file in result]

    # 按照第一个数值后缀排序
    suffixes.sort(key=lambda x: x[0])

    # 打开视频文件
    video_clip = VideoFileClip(video_file_path)

    # 遍历分割点，创建间隔视频
    for i in tqdm(range(len(suffixes) - 1), desc="创建间隔视频"):
        start_frame = suffixes[i][1]
        end_frame = suffixes[i + 1][1]

        # 计算开始和结束时间
        start_time = start_frame / video_clip.fps
        end_time = end_frame / video_clip.fps

        # 提取子视频
        subclip = video_clip.subclip(start_time, end_time)

        # 保存子视频
        output_path = os.path.join(output_folder, f"interval_{i}.mp4")
        subclip.write_videofile(output_path, codec='libx264')

    # 关闭视频文件
    video_clip.close()

def extract_numeric_suffix(filename):
    """
    使用正则表达式提取数值后缀
    """
    match = re.search(r'(\d+)\.\w+$', filename)
    if match:
        return int(match.group(1))
    return None

def process_video(video_file_path, no_of_frames=12, step=10, output_folder="interval_videos"):
    """
    处理视频文件，提取所有帧和关键帧，并找到与每个关键帧最相似的帧。然后根据最长递增子序列创建间隔视频。

    :param video_file_path: 视频文件的路径
    :param no_of_frames: 要提取的关键帧数量，默认为 12
    :param step: 每隔多少帧比较一次，默认为 10
    :param output_folder: 输出间隔视频的文件夹路径
    :return: 包含关键帧路径和最相似帧名称的列表
    """

    # 提取所有帧并保存
    all_frames_folder = "all_frames"
    ensure_folder_exists(all_frames_folder)
    extract_all_frames(video_file_path, all_frames_folder)

    # 提取关键帧
    temp_folder = extract_video_keyframes(video_file_path, no_of_frames)

    # 找到与每个关键帧最相似的帧
    result = []
    for keyframe_name in tqdm(os.listdir(temp_folder), desc="寻找最相似帧"):
        keyframe_path = os.path.join(temp_folder, keyframe_name)
        most_similar_frame_name = find_most_similar_frame(keyframe_path, all_frames_folder, step)
        result.append([keyframe_path, most_similar_frame_name])

    # 找到最长递增子序列
    longest_subseq_result = find_longest_increasing_subsequence(result)

    # 创建间隔视频
    ensure_folder_exists(output_folder)
    create_interval_videos(video_file_path, longest_subseq_result, output_folder)

    return longest_subseq_result

# 处理视频并创建间隔视频
result = process_video("BV14UD6YFEh1.mp4", no_of_frames=24, output_folder="BV14UD6YFEh1_interval_videos")
```

### 3. Run the Script

Save the script to a file, for example, `video_segmentation.py`, and run it:

```bash
python video_segmentation.py
```

### 4. download some videos to local 
```python
import os
import re
import shutil
from moviepy.editor import VideoFileClip

# 视频链接的文本
video_links_text = '''
【【原神·尘歌壶】翠黛峰一体化|扫冬峰 峰上人间 | 摹本分享】https://www.bilibili.com/video/BV14UD6YFEh1?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【【尘歌壶｜定制】翠黛峰一体化|抱秋峰 四方戏台 双区设计】https://www.bilibili.com/video/BV12YDaYME9Z?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【【尘歌壶｜定制】翠黛峰一体化|茶舍 捉夏峰 双区设计】https://www.bilibili.com/video/BV1UfDZYjE6R?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【【尘歌壶｜定制】翠黛峰一体化|数春峰 演武场 闲云vs钟离？】https://www.bilibili.com/video/BV1ibDQYkER5?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【『旦逢良辰，顺颂时宜——朱颜长似，岁岁年年｜献予生辰礼』】https://www.bilibili.com/video/BV1Ao18YjE2C?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【【尘歌壶｜定制】废土温室 枫丹壶三区露雾礁｜官服b服摹本分享】https://www.bilibili.com/video/BV1qrmNYTELp?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【【尘歌壶｜定制】集樱阁 稻妻壶一区霞见滩】https://www.bilibili.com/video/BV1HTmPYrE6r?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【【原神·尘歌壶】金色山脉下的童话小镇 罗浮洞四区燎雾岛带主宅】https://www.bilibili.com/video/BV1ur2oYNEXE?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【『One last kiss｜一分二十秒，献给自己的生日贺礼！』】https://www.bilibili.com/video/BV1FC1kYCErf?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
【【原神·尘歌壶】风车，落日，向日葵 | 娜维娅生贺壶】https://www.bilibili.com/video/BV1cKe4eWErc?vd_source=40cbe1b4dd51a83f3a39affb08e11e45
'''

video_links = re.findall(r'https://www\.bilibili\.com/video/BV\w+', video_links_text)

# 打印视频链接列表
print(video_links)

# 下载视频并重命名
for link in video_links:
    # 使用 BBDown.exe 下载视频
    os.system(f'BBDown.exe {link}')

    # 提取 BV 号
    bv_code = re.search(r'BV\w+', link).group(0)

    # 获取当前目录下的所有 .mp4 文件
    mp4_files = [f for f in os.listdir('.') if f.endswith('.mp4')]

    # 根据创建时间排序，选择最新的 .mp4 文件
    latest_file = max(mp4_files, key=lambda f: os.path.getctime(f))

    # 重命名文件
    new_filename = f'{bv_code}.mp4'
    clip = VideoFileClip(latest_file)
    clip.write_videofile(new_filename)
    print(f'Renamed {latest_file} to {new_filename}')

```

### 5. mp4 downloaded split in iter 
```python
import pathlib
import pandas as pd
import numpy as np
mp4_l = pd.Series(list(pathlib.Path(".").rglob("BV*.mp4"))).map(str).map(
    lambda x: np.nan if "\\" in x else x
).dropna().values.tolist()
mp4_l

from tqdm import tqdm
from IPython import display
for mp4_path in tqdm(mp4_l):
    name = mp4_path.split(".")[0]
    result = process_video(mp4_path, no_of_frames=24, output_folder="{}_interval_videos".format(name))
    display.clear_output(wait = True)
```

### 6. mp4 to image frames 
```python
import pathlib
import pandas as pd
import numpy as np

video_path_df = pd.DataFrame(
pd.Series(list(pathlib.Path(".").rglob("BV*_interval_videos"))).map(str).map(
    lambda vp: pd.Series(list(pathlib.Path(vp).rglob("*.mp4"))).map(str).values.tolist()
).explode().dropna().map(
    lambda x: (x, x.replace("\\", "_").replace(".mp4", ""))
).values.tolist())
video_path_df.columns = ["mp4_path", "frame_path"]
video_path_df


import os
from moviepy.editor import VideoFileClip
from PIL import Image
from tqdm import tqdm

def extract_frames(video_path, output_folder, max_images_per_folder=1000, max_folder_count = 2):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频文件
    clip = VideoFileClip(video_path)

    # 获取视频的总帧数
    total_frames = int(clip.fps * clip.duration)

    # 初始化变量
    frame_count = 0
    folder_count = 0

    # 创建第一个子文件夹
    current_folder = os.path.join(output_folder, f"{output_folder}_folder_{folder_count}")
    os.makedirs(current_folder, exist_ok=True)

    # 遍历每一帧
    for frame_idx, frame in tqdm(enumerate(clip.iter_frames())):
        # 如果当前文件夹中的图片数量达到上限，创建新的文件夹
        if frame_count >= max_images_per_folder:
            if folder_count >= max_folder_count:
                break
            folder_count += 1
            current_folder = os.path.join(output_folder, f"{output_folder}_folder_{folder_count}")
            os.makedirs(current_folder, exist_ok=True)
            frame_count = 0

        # 保存当前帧为图片
        frame_filename = os.path.join(current_folder, f"frame_{frame_count:06d}.png")
        Image.fromarray(frame).save(frame_filename)

        # 更新帧计数器
        frame_count += 1

    print(f"Extracted {total_frames} frames into {folder_count + 1} folders.")

'''
from IPython import display
for i, r in tqdm(video_path_df.iterrows()):
    mp4_path = r["mp4_path"]
    frame_path = r["frame_path"]
    extract_frames(mp4_path, frame_path, max_images_per_folder=200, max_folder_count = 2)
    display.clear_output(wait=True)
'''

from tqdm import tqdm
from IPython import display
from concurrent.futures import ThreadPoolExecutor

def process_video(row):
    mp4_path = row["mp4_path"]
    frame_path = row["frame_path"]
    extract_frames(mp4_path, frame_path, max_images_per_folder=200, max_folder_count=2)
    display.clear_output(wait=True)

# 设置并行数
parallel_count = 12  # 你可以根据需要调整这个值

with ThreadPoolExecutor(max_workers=parallel_count) as executor:
    list(tqdm(executor.map(process_video, 
                           list(map(lambda t2: t2[1] ,video_path_df.iterrows()))
                          ), total=len(video_path_df)))
```

## References

- [Katna GitHub Repository](https://github.com/keplerlab/Katna)
- [Katna Documentation](https://katna.readthedocs.io/en/latest/tutorials_video.html)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Description

Katna automates the boring, error prone task of videos key/best frames extraction,
video compression and manual time consuming task of image cropping and resizing using ML.

In short, you may want to consider using Katna library if you have following tasks:

1. You have video/videos from who you want to extract keyframe/keyframes. 
   Please note Key-frames are defined as the representative frames of a video stream,
   the frames that provide the most accurate and compact summary of the video content.
   Take an example of this video and some of the top keyframes extracted using Katna. 
   
[![IMAGE ALT TEXT HERE](docs/source/images/tide_video_thumnail.jpg)](https://www.youtube.com/watch?v=zpaLHwwYxE8)

<p align="center"><img src="docs/source/images/arrow_down.jpeg" alt="arrow down" width="20"/></p>

![Image of keyframe extracted using Katna](docs/source/images/keyframe_extraction.jpg "Image of keyframe extracted using Katna")


2. You have video/videos you want to compress down to smaller size. (e.g. You have video with let's say 1 GB Size but you want to compress it down as small as possible.)

3. You have image/images which you want to smartly resize to a target resolution.
   (e.g. 500x500, 1080p (1920x1080) etc.)

![Katna Image resize](docs/source/images/katna_image_resize.jpg "Katna Image resize")

4. You have image/images from which you want to intelligently extract a crop with a target resolution.
   (e.g. Get a crop of size 500x500 from image of size 1920x1080)

![Katna Image crop](docs/source/images/katna_image_crop.jpg "Katna Image crop")


5. You want to extract a crop of particular aspect ratio e.g. 4:3 from your input image/images.
   (e.g. Get a crop of aspect ratio 1:1 from image of resolution 1920x1080 (16:9 aspect ratio image))

6. You want to resize a video to particular aspect ratio e.g. 16:9 (Landscape), to lets say to 1:1 (square). Please note that this feature is currently **experimental** and needs additional compiling and configuration of google [mediapipe library](https://github.com/google/mediapipe). 


[![ICEGov original video with 16:9 aspect ratio](docs/source/images/icegov_original.jpeg)](https://www.youtube.com/watch?v=-GFabrw3Csk)

<p align="center"><img src="docs/source/images/arrow_down.jpeg" alt="arrow down" width="20"/></p>

[![ICEGov resized video with 1:1 aspect ratio](docs/source/images/icegov_1_1_thumbnail.jpeg)](https://www.youtube.com/watch?v=P0D5WPv63RY)


Katna is divided into two modules
* Video module.
* Image module.

Video Module:
-------------
This module handles the task(s) for key frame(s) extraction and video compression.

Key-frames are defined as the representative frames of a video stream, the frames that provide the most accurate and compact summary of the video content.

**Frame extraction and selection criteria for key-frame extraction**

1. Frame that are sufficiently different from previous ones using absolute differences in LUV colorspace
2. Brightness score filtering of extracted frames
3. Entropy/contrast score filtering of extracted frames
4. K-Means Clustering of frames using image histogram
5. Selection of best frame from clusters based on and variance of laplacian (image blur detection)

Video compression is handled using ffmpeg library. Details about which could be read in [Katna.video_compressor module](https://katna.readthedocs.io/en/latest/understanding_katna.html#katna-video-compressor) section.

Since version 0.8.0 of Katna we are extending smart resize features to videos with the help of Google's Mediapipe project. To know more about this please refer to documentation [Video Smart Resize using Katna]
(https://katna.readthedocs.io/en/latest/understanding_katna.html#katna-video-resize). Please note that this feature is an optional experimental feature. And might be subject to removal/modifications at later versions. Also you also need to install Google's Mediapipe library, Specially autoflip binary for this to work. Please refer to [Link](https://katna.readthedocs.io/en/latest/tutorials_video_smart_resize.html#tutorials-video-smart-resize) for how to install and configure mediapipe to be used with katna. 

Image Module:
-------------
This module handles the task(s) related to smart cropping and image resizing.

The Smart image cropping is happening in way that the module identifies the best part or the area where someone focus more
and interprets this information while cropping the image.

**Crop extraction and selection criteria**

1. Edge, saliency and Face detection features are detected in the input image
2. All the crops with specified dimensions are extracted with calculation of score for each crop wrt to extracted features
3. The crops will be passes through filters specified which will remove the crops which filter rejects

Similar to Smart crop Katna image module supports **Smart image resizing** feature. Given an input image it can resize image to target resolution with simple resizing if aspect ratio is same for input and target image. If aspect ratio is different than smart image resize will first crops biggest good quality crop in target resolution and then resizes image in target resolution. This ensures image resize without actually skewing input image. *Please not that if aspect ratio of input and output image are not same katna image_resize can lead to some loss of image content*

**Supported Video and image file formats**
##########################################

All the major video formats like .mp4,.mov,.avi etc and image formats like .jpg, .png, .jpeg etc are supported. 

More selection features are in developement pipeline

###  How to install

#### Using pypi
1) Install Python 3 
2) pip install katna

#### Install from source

1) Install git
2) Install Anaconda or Miniconda Python
3) Open terminal 
4) Clone repo from here https://github.com/keplerlab/Katna.git 
5) Change the directory to the directory where you have cloned your repo 
    ```
    $cd path_to_the_folder_repo_cloned
    ```
6) Create a new anaconda environment if you are using anaconda python distribution
    ```
    conda create --name katna python=3.7
    source activate katna
    ```

7) Run the setup:
    ``` 
    python setup.py install 
    ```    

#### Error handling and updates 
1) Since Katna version 0.4.0 Katna video module is optimized to use multiprocessing using python multiprocessing module. Due to restrictions of multiprocessing in windows, For safe importing of main module in windows system, make sure “entry point” of the program is wrapped in  __name__ == '__main__': as follows:
    ```
    from Katna.video import Video
    if __name__ == "__main__":
        vd = Video()
        # your code
    ```
    please refer to https://docs.python.org/2/library/multiprocessing.html#windows for more details.  

2) If input image is of very large size ( larger than 2000x2000 ) it might take a
long time to perform Automatic smart cropping.If you encounter this issue, consider changing down_sample_factor
from default 8 to larger values ( like 16 or 32 ). This will decrease processing time 
significantly. 

3) If you see "AttributeError: module 'cv2.cv2' has no attribute 'saliency'" error. Uninstall opencv-contrib
by running command "python -m pip uninstall opencv-contrib-python" and then again install it by running command 
    ```
    python -m pip install opencv-contrib-python
    ```

4) If you see "FileNotFoundError: frozen_east_text_detection.pb file not found". Open python shell and follow below commands.
    ```
    from Katna.image_filters.text_detector import TextDetector
    td = TextDetector()
    td.download()
    ```

5) On windows, ensure that anaconda has admin rights if installing with anaconda as it fails with 
the write permission while installing some modules.

6) If you get "RuntimeError: No ffmpeg exe could be found". Install ffmpeg on your system, and/or set the IMAGEIO_FFMPEG_EXE or FFMPEG_EXE environment variable to path of your ffmpeg binary.
Usually ffmpeg is installed using imageio-ffmpeg package, Check **imageio_ffmpeg-*.egg** folder inside your
**site-packages** folder, there should be a ffmpeg file inside binaries folder, check if this file has proper read/executable permission set and additionally set it's path to environment variable.

7) There is a known memory leak issue in Katna version 0.8.2 and less,
    when running bulk video keyframe extraction on Python version 3.6 and 3.7, 
    This is an multiprocessing bug observed only in Python 3.6 and 3.7. And is fixed in katna version 0.9 and above. If you are running Keyframe extraction code on large number of videos and facing memory issue, request you to upgrade your katna version to version 0.9 or above. If you still want to use older version of katna consider upgrading your python version to 3.8 or above.
### How to use Library

1) Refer to quickstart section in Katna Reference 
   from https://katna.readthedocs.io/

### Update: katna version 0.9.0
We have added writer framework to process data from Katna Video and Image module. This version
also fixes memory leak issue reported by [this](https://github.com/keplerlab/katna/issues/11) and
[this](https://github.com/keplerlab/katna/issues/12) issue.

#### The version introduces following breaking changes in the library API: ####
1. video.extract_video_keyframes and video.extract_video_keyframes_from_dir requires additional writer object. By default, KeyFrameDiskWriter is available to use from
Katna.writer module. Writer framework can be extended based on the requirement.

2. image.crop_image and image.crop_image_from_dir requires additional writer object.
By default, ImageCropDiskWriter is available to use from
Katna.writer module. Writer framework can be extended based on the requirement.

Refer documentation for the updated examples here: [Link](https://katna.readthedocs.io/)

### Update: katna version 0.8.2
This bug fix version fixes this issue: [Link](https://github.com/keplerlab/katna/issues/10)
### Update: katna version 0.8.1
Fixed an issue where in rare case where videos split using ffmpeg not readable and throwing exception [Link](https://github.com/keplerlab/katna/issues/9)
### Update: katna version 0.8.0
Added experimental support for autocrop/resize videos using Google's mediapipe
Autoflip code example.
### Update: katna version 0.7.1
Fixed bug where incorrect specification crops were returned by image_module crop_image and crop_image_from_dir method. 
### Update: katna version 0.7.0
Added support for video compression in Katna.video module.
### Update: katna version 0.6.0
Added support for smart image resize in Katna.image module.
### Update: katna version 0.5.0
In version 0.5.0 we have changed name of some of the public functions inside
for Katna.video module used for keyframe extraction,
1) extract_frames_as_images method is changed to extract_video_keyframes.
2) extract_frames_as_images_from_dir method is changed to extract_keyframes_from_videos_dir
### Attributions
1) We have used the SAD (Sum of absolute difference) code from [KeyFramesExtraction](https://github.com/amanwalia92/KeyFramesExtraction) project by Amanpreet Walia. Code released under MIT Licence.
2) We have used project [smartcrop.js](https://github.com/jwagner/smartcrop.js/) for Smart crop feature in Katna Image module.
3) For Experimental feature of Smartcrop/Resize in videos we are using help of [Google Mediapipe](https://github.com/google/mediapipe) [Autoflip](https://ai.googleblog.com/2020/02/autoflip-open-source-framework-for.html) framework.
4) Katna icon generated by [thenounproject](https://thenounproject.com/term/chef-knife/2082763/) icon developed by ProSymbols, US , In the Viking Elements Glyph Icons Collection licensed as Creative Commons CCBY.
