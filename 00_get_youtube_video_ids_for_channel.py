import scrapetube

videos = scrapetube.get_channel("UCvKRFNawVcuz4b9ihUTApCg")

for video in videos:
    with open('video_ids.txt', 'a', encoding='utf-8') as f:
        f.write(video['videoId'] + '\n')
