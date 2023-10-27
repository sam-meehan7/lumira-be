import json
import os
from pytube import YouTube
import moviepy.editor as mp

# Initialize an empty dictionary to store video details
video_dict = {}

# Create a data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')


def download_audio(video_ids):
    for video_id in video_ids:
        url = f'https://www.youtube.com/watch?v={video_id}'

        try:
            yt = YouTube(url)

            # Access and print metadata
            print(f'Title: {yt.title}')
            print(f'Author: {yt.author}')
            print(f'Description: {yt.description}')
            print(f'Views: {yt.views}')

            # Store metadata in video_dict
            video_dict[video_id] = {'title': yt.title, 'url': url}

            # Try to get the mp4 audio stream
            audio_stream = yt.streams.filter(
                only_audio=True, file_extension='mp4'
            ).first()

            # Define the output path within the 'data' directory
            output_path = os.path.join('data', f'{video_id}')

            if audio_stream:
                # Download the mp4 audio stream
                audio_stream.download(
                    output_path=output_path, filename=f'{video_id}.mp4'
                )

                # Convert mp4 to mp3
                audio_clip = mp.AudioFileClip(
                    os.path.join(output_path, f'{video_id}.mp4')
                )
                audio_clip.write_audiofile(
                    os.path.join(output_path, f'{video_id}.mp3'), codec='mp3'
                )

                print(
                    f'MP4 audio downloaded and converted to mp3 for video id: {video_id}'
                )
            else:
                # If mp4 is not available, try webm
                audio_stream = yt.streams.filter(
                    only_audio=True, file_extension='webm'
                ).first()

                if audio_stream:
                    # Download the webm audio stream
                    audio_stream.download(
                        output_path=output_path, filename=f'{video_id}.webm'
                    )

                    # Convert webm to mp3
                    audio_clip = mp.AudioFileClip(
                        os.path.join(output_path, f'{video_id}.webm')
                    )
                    audio_clip.write_audiofile(
                        os.path.join(output_path, f'{video_id}.mp3'), codec='mp3'
                    )

                    print(
                        f'WebM audio downloaded and converted to mp3 for video id: {video_id}'
                    )
                else:
                    print(f'No audio stream found for video id: {video_id}')
        except Exception as e:
            print(f'An error occurred while processing video id {video_id}: {e}')

    # Write video_dict to a JSON file within the 'data' directory
    with open(os.path.join('data', 'video_dict.json'), 'w') as json_file:
        json.dump(video_dict, json_file)


# Read video IDs from a text file and store them in a list
with open('video_ids.txt', 'r') as file:
    video_ids = [line.strip() for line in file]

# Call the function to download audio
download_audio(video_ids)
