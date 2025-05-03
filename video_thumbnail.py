import cv2
import math
from PIL import Image
import os
import subprocess
import tempfile
from pathlib import Path
import glob

# Configuration
THUMBNAIL_WIDTH = 640
THUMBNAIL_HEIGHT = 480
FRAMES_PER_ROW = 60
OUTPUT_IMAGE = 'video_thumbnail.jpg'


def find_dvd_mount():
    """Find DVD mount point in Chrome OS Linux container."""
    # Check common Chrome OS mount points
    mount_points = [
        '/mnt/chromeos/removable/*/VIDEO_TS',
        '/media/removable/*/VIDEO_TS'
    ]
    
    for pattern in mount_points:
        matches = glob.glob(pattern)
        if matches:
            return os.path.dirname(matches[0])  # Return the parent directory of VIDEO_TS
    return None


def is_dvd_device(path):
    """Check if the given path is a DVD device or mount point."""
    if path.startswith('/dev/dvd') or path.startswith('/dev/sr'):
        return True
    if os.path.exists(os.path.join(path, 'VIDEO_TS')):
        return True
    return False


def extract_dvd_to_temp(video_path):
    """Extract DVD content to a temporary file using ffmpeg."""
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, 'dvd_content.mp4')
    
    try:
        # If it's a mount point, use the VIDEO_TS directory
        if os.path.isdir(video_path):
            video_ts_path = os.path.join(video_path, 'VIDEO_TS')
            if not os.path.exists(video_ts_path):
                print(f"ERROR: VIDEO_TS directory not found at {video_ts_path}")
                return None
            
            # Find all VOB files
            vob_files = glob.glob(os.path.join(video_ts_path, 'VTS_*_[0-9].VOB'))
            if not vob_files:
                print("ERROR: No VOB files found in VIDEO_TS directory")
                return None
            
            # Get the largest VOB file (usually the main movie)
            largest_vob = max(vob_files, key=os.path.getsize)
            print(f"Using VOB file: {largest_vob}")
            video_path = largest_vob
        
        # Use ffmpeg to read from DVD and convert to MP4
        cmd = [
            'ffmpeg', '-i', video_path,
            '-c:v', 'libx264', '-crf', '23',
            '-preset', 'fast',
            output_path
        ]
        print(f"Running ffmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(output_path):
            return output_path
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error extracting DVD content: {e}")
        if e.stderr:
            print(f"ffmpeg error output: {e.stderr}")
        return None


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
    parser.add_argument('video', help='Path to the video file, DVD device, or DVD mount point')
    parser.add_argument('--output', default=OUTPUT_IMAGE, help='Output image file name')
    args = parser.parse_args()

    video_path = args.video
    temp_file = None

    try:
        if is_dvd_device(video_path):
            print(f"Detected DVD device or mount point: {video_path}")
            print("Extracting DVD content...")
            dvd_mount = find_dvd_mount()
            if dvd_mount:
                video_path = dvd_mount
            temp_file = extract_dvd_to_temp(video_path)
            if not temp_file:
                print("Failed to extract DVD content")
                return
            video_path = temp_file

        print(f"Extracting frames from {video_path}...")
        frames = extract_one_frame_per_second(video_path)
        print(f"Extracted {len(frames)} frames.")

        print("Creating composite thumbnail...")
        composite = make_composite_thumbnail(frames)
        if composite:
            composite.save(args.output)
            print(f"Thumbnail saved as {args.output}")
        else:
            print("No frames extracted. Thumbnail not created.")
    finally:
        # Clean up temporary file if it was created
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            os.rmdir(os.path.dirname(temp_file))


if __name__ == "__main__":
    main() 
