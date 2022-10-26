import os, sys
import json 
import pandas as pd
import time 

from pytube import YouTube

def download_video(video_url, video_name):
    yt = YouTube(video_url) 
    mp4_files = yt.streams.filter(file_extension="mp4")
    mp4_369p_files = mp4_files.get_by_resolution("360p")
    mp4_369p_files.download(f"data/v", filename=f'{video_name}.mp4')
    time.sleep(5)

url = 'https://www.youtube.com/watch?v=9kyPX1qZbfY'
download_video(url, 'quantumania')
# base_url = 'https://www.youtube.com/watch?v={}'

# video_id_csv = 'data/files/yttemporal1b_ids_val.csv'

# video_id_df = pd.read_csv(video_id_csv)
# print(video_id_df.head())   

# video_ids = video_id_df['video_id'].tolist()
# print(video_ids[:5])

# for vidx, video_id in enumerate(video_ids[:100]):
#     print(f'downloading vidx:{vidx+1}/100')

#     video_url = base_url.format(video_id)
#     yt = YouTube(video_url) 
#     mp4_files = yt.streams.filter(file_extension="mp4")
#     mp4_369p_files = mp4_files.get_by_resolution("360p")
#     mp4_369p_files.download(f"data/v", filename=f'{video_id}.mp4')
#     time.sleep(5)