import json
from pytube import YouTube
import moviepy.editor as mp
import os

video_dict = {}


def download_audio(video_ids):
    for video_id in video_ids:
        url = f'https://www.youtube.com/watch?v={video_id}'

        try:
            yt = YouTube(url)

            # Accessing metadata
            print(f'Title: {yt.title}')
            print(f'Author: {yt.author}')
            print(f'Description: {yt.description}')

            print(f'Views: {yt.views}')

            video_dict[video_id] = {}
            video_dict[video_id]['title'] = yt.title
            video_dict[video_id]['url'] = url

            # First, try to get the mp4 audio stream
            audio_stream = yt.streams.filter(
                only_audio=True, file_extension='mp4'
            ).first()
            if audio_stream:
                # Download the audio stream
                output_path = audio_stream.download(filename=f'{video_id}')
                # Get the filename from the output path
                filename = os.path.basename(output_path)
                # Convert the audio to mp3
                audio_clip = mp.AudioFileClip(filename)
                audio_clip.write_audiofile(
                    f'{os.path.splitext(filename)[0]}.mp3', codec='mp3'
                )
                print(
                    f'MP4 audio downloaded and converted to mp3 for video id: {video_id}'
                )
            else:
                # If mp4 audio stream is not available, try to get the webm audio stream
                audio_stream = yt.streams.filter(
                    only_audio=True, file_extension='webm'
                ).first()
                if audio_stream:
                    # Download the audio stream
                    output_path = audio_stream.download(filename=f'{video_id}')
                    # Get the filename from the output path
                    filename = os.path.basename(output_path)
                    # Convert the audio to mp3
                    audio_clip = mp.AudioFileClip(filename)
                    audio_clip.write_audiofile(
                        f'{os.path.splitext(filename)[0]}.mp3', codec='mp3'
                    )
                    print(
                        f'WebM audio downloaded and converted to mp3 for video id: {video_id}'
                    )
                else:
                    print(f'No audio stream found for video id: {video_id}')
        except Exception as e:
            print(f'An error occurred while processing video id {video_id}: {e}')

    # Write video_dict to a json file
    with open('video_dict.json', 'w') as json_file:
        json.dump(video_dict, json_file)


# List of YouTube video ids
video_ids = ["d-p1LxIIkiA", "oLZQ6LsUgkg", "LfI1SdbZbdY"]

# # Call the function to download audio
download_audio(video_ids)
