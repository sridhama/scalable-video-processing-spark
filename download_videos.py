import os, sys
import json 
import pandas as pd
import time 
import pandas as pd 

from pytube import YouTube

def download_video_helper(video_url, video_name):
    yt = YouTube(video_url) 
    mp4_files = yt.streams.filter(file_extension="mp4")
    mp4_369p_files = mp4_files.get_by_resolution("360p")
    mp4_369p_files.download(f"data/v", filename=f'{video_name}.mp4')
    time.sleep(5)

# url = 'https://www.youtube.com/watch?v=9kyPX1qZbfY'
# download_video_helper(url, 'quantumania')

def download_videos():
    video_id_csv = 'data/files/yttemporal1b_ids_val.csv'
    df = pd.read_csv(video_id_csv)
    print(df.head())

    video_ids = df['video_id'].tolist()
    print('total number of videos',len(video_ids))

    # Set videos_to_download=x to download x number of videos. Default:10
    videos_to_download = 1000

    downloaded = 0 
    downloaded_max = 50

    for i in range(videos_to_download+100):
    to_path = f'data/v/{video_ids[i]}.mp4'
    if os.path.exists(to_path):
        downloaded += 1
        print(f'video: {video_ids[i]} already downloaded. total_downloaded: {downloaded}')
        continue
    url = f'https://www.youtube.com/watch?v={video_ids[i]}'
    try:
        download_video_helper(url, f'{video_ids[i]}')
        downloaded += 1
        print(f'downloaded {i+1} {video_ids[i]}. total_downloaded: {downloaded}')
    except:
        print(f'video i: {i} {video_ids[i]} not downloaded')
    if downloaded == downloaded_max:
        break