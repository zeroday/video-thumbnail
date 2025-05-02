import cv2
import math
from PIL import Image
import os

# Configuration
THUMBNAIL_WIDTH = 320
THUMBNAIL_HEIGHT = 180
FRAMES_PER_ROW = 60
OUTPUT_IMAGE = 'video_thumbnail.jpg'


def extract_one_frame_per_second(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)
    frames = []
    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        frames.append(img)
    cap.release()
    return frames


def make_composite_thumbnail(frames, thumb_per_row=FRAMES_PER_ROW):
    if not frames:
        return None
    w, h = frames[0].size
    rows = math.ceil(len(frames) / thumb_per_row)
    composite = Image.new('RGB', (w * thumb_per_row, h * rows))
    for idx, frame in enumerate(frames):
        x = (idx % thumb_per_row) * w
        y = (idx // thumb_per_row) * h
        composite.paste(frame, (x, y))
    return composite


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate a composite video thumbnail.')
    parser.add_argument('video', help='Path to the video file')
    parser.add_argument('--output', default=OUTPUT_IMAGE, help='Output image file name')
    args = parser.parse_args()

    print(f"Extracting frames from {args.video}...")
    frames = extract_one_frame_per_second(args.video)
    print(f"Extracted {len(frames)} frames.")

    print("Creating composite thumbnail...")
    composite = make_composite_thumbnail(frames)
    if composite:
        composite.save(args.output)
        print(f"Thumbnail saved as {args.output}")
    else:
        print("No frames extracted. Thumbnail not created.")


if __name__ == "__main__":
    main() 