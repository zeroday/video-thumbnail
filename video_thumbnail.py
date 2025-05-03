import cv2
import math
from PIL import Image
import os
import subprocess
import tempfile
from pathlib import Path
import glob
import logging
import sys
import traceback
from datetime import datetime
import re
import numpy as np
from typing import List, Tuple

# Set up logging
log_filename = f'video_thumbnail_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
THUMBNAIL_WIDTH = 320
THUMBNAIL_HEIGHT = 180  # 320 / (16/9) ≈ 180
FRAMES_PER_ROW = 60
OUTPUT_IMAGE = 'video_thumbnail.jpg'
CHECKPOINT_FILE = 'thumbnail_checkpoint.npz'
BATCH_SIZE = 50  # Reduced from 100 to 50 for even better memory management
MAX_FRAMES_IN_MEMORY = 500  # Reduced from 1000 to 500 for more conservative memory usage


def find_dvd_mount():
    """Find DVD mount point in Chrome OS Linux container."""
    logger.info("Searching for DVD mount points...")
    # Check common Chrome OS mount points
    mount_points = [
        '/mnt/chromeos/removable/*/VIDEO_TS',
        '/media/removable/*/VIDEO_TS'
    ]
    
    for pattern in mount_points:
        matches = glob.glob(pattern)
        if matches:
            mount_point = os.path.dirname(matches[0])
            logger.info(f"Found DVD mount point: {mount_point}")
            return mount_point
    logger.warning("No DVD mount points found")
    return None


def is_dvd_device(path):
    """Check if the given path is a DVD device or mount point."""
    logger.debug(f"Checking if {path} is a DVD device")
    if path.startswith('/dev/dvd') or path.startswith('/dev/sr'):
        logger.info(f"Found DVD device: {path}")
        return True
    if os.path.exists(os.path.join(path, 'VIDEO_TS')):
        logger.info(f"Found DVD mount point: {path}")
        return True
    logger.debug(f"{path} is not a DVD device")
    return False


def parse_duration(duration_str):
    """Parse duration string in format 'HH:MM:SS' or 'MM:SS' into seconds."""
    parts = duration_str.split(':')
    if len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:  # MM:SS
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid duration format: {duration_str}")


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
        logger.error(f"ffprobe failed: {result.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        return None


def analyze_vob_files(video_ts_path, expected_duration=None):
    """Analyze VOB files to determine the main movie and its properties."""
    logger.info(f"Analyzing VOB files in {video_ts_path}")
    
    # Find all VOB files
    vob_files = glob.glob(os.path.join(video_ts_path, 'VTS_*_[0-9].VOB'))
    if not vob_files:
        logger.error("No VOB files found")
        return None
    
    # Get IFO files for analysis
    ifo_files = glob.glob(os.path.join(video_ts_path, 'VTS_*_0.IFO'))
    if not ifo_files:
        logger.error("No IFO files found")
        return None
    
    # Analyze each IFO file to find the main movie
    main_movie_info = None
    for ifo_file in ifo_files:
        # Get the VTS number from the IFO filename
        vts_num = os.path.basename(ifo_file).split('_')[1]
        logger.debug(f"Analyzing VTS {vts_num}")
        
        # Get corresponding VOB files
        vts_vobs = [f for f in vob_files if f'VTS_{vts_num}_' in f]
        if not vts_vobs:
            logger.warning(f"No VOB files found for VTS {vts_num}")
            continue
        
        # Calculate total size of VOB files
        total_size = sum(os.path.getsize(f) for f in vts_vobs)
        
        # Try to get duration from the first VOB file
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                vts_vobs[0]
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                logger.info(f"VTS {vts_num} duration: {duration/60:.1f} minutes")
                
                # If expected duration is provided, check if this title matches
                if expected_duration is not None:
                    duration_diff = abs(duration - expected_duration)
                    if duration_diff > 3600:  # More than 1 hour difference
                        logger.warning(f"VTS {vts_num} duration ({duration/60:.1f} min) differs significantly from expected duration ({expected_duration/60:.1f} min)")
                        if duration > expected_duration * 2:  # More than twice the expected duration
                            logger.error(f"VTS {vts_num} duration is significantly longer than expected. Skipping this title.")
                            continue
                
                # Main movie is usually the largest set of VOB files with matching duration
                if main_movie_info is None or (total_size > main_movie_info['total_size'] and 
                                             (expected_duration is None or 
                                              abs(duration - expected_duration) < abs(main_movie_info['duration'] - expected_duration))):
                    main_movie_info = {
                        'ifo_file': ifo_file,
                        'vts_num': vts_num,
                        'vob_files': vts_vobs,
                        'total_size': total_size,
                        'duration': duration
                    }
        except Exception as e:
            logger.warning(f"Could not determine duration for VTS {vts_num}: {str(e)}")
    
    if main_movie_info:
        logger.info(f"Main movie found: VTS {main_movie_info['vts_num']}")
        logger.info(f"Number of VOB files: {len(main_movie_info['vob_files'])}")
        logger.info(f"Total VOB size: {main_movie_info['total_size']/1024/1024:.1f} MB")
        logger.info(f"Estimated duration: {main_movie_info['duration']/60:.1f} minutes")
        
        # Final duration validation if expected duration is provided
        if expected_duration is not None:
            duration_diff = abs(main_movie_info['duration'] - expected_duration)
            if duration_diff > 3600:  # More than 1 hour difference
                logger.warning(f"Main movie duration ({main_movie_info['duration']/60:.1f} min) differs significantly from expected duration ({expected_duration/60:.1f} min)")
                if main_movie_info['duration'] > expected_duration * 2:  # More than twice the expected duration
                    logger.error("Main movie duration is significantly longer than expected. This might not be the correct title.")
                    return None
        
        return main_movie_info
    else:
        logger.error("Could not determine main movie")
        return None


def extract_dvd_to_temp(video_path, expected_duration=None):
    """Extract DVD content to a temporary file using ffmpeg."""
    logger.info(f"Starting DVD extraction from {video_path}")
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Created temporary directory: {temp_dir}")
    output_path = os.path.join(temp_dir, 'dvd_content.mp4')
    file_list = None
    
    try:
        # If it's a mount point, use the VIDEO_TS directory
        if os.path.isdir(video_path):
            video_ts_path = os.path.join(video_path, 'VIDEO_TS')
            logger.debug(f"Checking VIDEO_TS directory: {video_ts_path}")
            if not os.path.exists(video_ts_path):
                logger.error(f"VIDEO_TS directory not found at {video_ts_path}")
                return None
            
            # Analyze VOB files before extraction
            movie_info = analyze_vob_files(video_ts_path, expected_duration)
            if not movie_info:
                return None
            
            # Check for NVIDIA GPU support
            nvidia_supported = False
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                nvidia_supported = result.returncode == 0
                if nvidia_supported:
                    logger.info("NVIDIA GPU detected, will use hardware acceleration")
            except Exception:
                logger.info("No NVIDIA GPU detected, using software encoding")
            
            # Create concatenation string for VOB files
            vob_files = sorted(movie_info['vob_files'])
            concat_string = "concat:" + "|".join(vob_files)
            logger.debug(f"Using concatenation string: {concat_string}")
            
            # Build ffmpeg command with appropriate options
            cmd = ['ffmpeg', '-i', concat_string]
            
            # Add hardware acceleration if available
            if nvidia_supported:
                cmd.extend([
                    '-hwaccel', 'cuda',
                    '-hwaccel_device', '0',
                    '-hwaccel_output_format', 'cuda',
                    '-c:v', 'h264_nvenc'
                ])
            else:
                cmd.extend(['-c:v', 'libx264'])
            
            # Add deinterlacing and other options
            cmd.extend([
                '-vf', 'yadif=1',  # Deinterlacing
                '-crf', '23',
                '-preset', 'fast',
                output_path
            ])
            
            logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg failed with return code {result.returncode}")
                logger.error(f"ffmpeg stderr: {result.stderr}")
                return None
            
            if os.path.exists(output_path):
                logger.info(f"Successfully created output file: {output_path}")
                
                # Validate duration if expected duration is provided
                if expected_duration is not None:
                    actual_duration = get_video_duration(output_path)
                    if actual_duration is not None:
                        duration_diff = abs(actual_duration - expected_duration)
                        if duration_diff > 3600:  # More than 1 hour difference
                            logger.warning(f"Extracted video duration ({actual_duration/60:.1f} min) differs significantly from expected duration ({expected_duration/60:.1f} min)")
                            
                            # Try to fix the duration by re-encoding with duration constraint
                            fixed_path = os.path.join(temp_dir, 'dvd_content_fixed.mp4')
                            cmd = [
                                'ffmpeg', '-i', concat_string,
                                '-c:v', 'libx264',
                                '-vf', 'yadif=1',
                                '-crf', '23',
                                '-preset', 'fast',
                                '-t', str(expected_duration),
                                fixed_path
                            ]
                            logger.info(f"Attempting to fix duration with command: {' '.join(cmd)}")
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if result.returncode == 0 and os.path.exists(fixed_path):
                                logger.info("Successfully created fixed duration video")
                                os.remove(output_path)
                                os.rename(fixed_path, output_path)
                            else:
                                logger.error("Failed to fix video duration")
                
                return output_path
            logger.error("Output file was not created")
            return None
            
    except Exception as e:
        logger.error(f"Error during DVD extraction: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    finally:
        # Clean up temporary files
        if file_list and os.path.exists(file_list):
            os.remove(file_list)
            logger.debug(f"Removed temporary file list: {file_list}")


def get_thumbnail_dimensions(size='default'):
    """Get thumbnail dimensions based on size parameter."""
    if size == 'xl':
        width = 640
    else:  # default
        width = 320
    height = int(width / (16/9))  # Maintain 16:9 aspect ratio
    return width, height


def warn_xl_size(duration: float):
    """Warn user about potential memory issues with XL size."""
    if duration > 3600:  # More than 1 hour
        logger.warning("Using XL size for videos longer than 1 hour may cause memory issues.")
        logger.warning("Consider using default size for very long videos.")
        logger.warning(f"Video duration: {duration/60:.1f} minutes")
        logger.warning("Processing in very small batches (50 frames) to manage memory usage")
        logger.warning("Maximum frames in memory: 500")


def save_checkpoint(frames: List[np.ndarray], processed_count: int):
    """Save checkpoint of processed frames."""
    try:
        np.savez_compressed(CHECKPOINT_FILE, frames=frames, count=processed_count)
        logger.info(f"Saved checkpoint with {processed_count} frames")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def load_checkpoint() -> Tuple[List[np.ndarray], int]:
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            data = np.load(CHECKPOINT_FILE, allow_pickle=True)
            frames = data['frames'].tolist()
            count = int(data['count'])
            logger.info(f"Loaded checkpoint with {count} frames")
            return frames, count
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    return [], 0


def process_frames_in_batches(cap: cv2.VideoCapture, duration: float, width: int, height: int) -> List[np.ndarray]:
    """Process frames in batches to manage memory usage."""
    frames = []
    current_second = 0
    
    while current_second < duration:
        # Set position to current second
        cap.set(cv2.CAP_PROP_POS_MSEC, current_second * 1000)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame at {current_second} seconds")
            current_second += 1
            continue
            
        # Resize frame
        frame = cv2.resize(frame, (width, height))
        frames.append(frame)
        
        # Save checkpoint and clear memory if we have too many frames
        if len(frames) % BATCH_SIZE == 0:
            save_checkpoint(frames, current_second)
            logger.info(f"Saved checkpoint with {len(frames)} frames")
            
        if len(frames) >= MAX_FRAMES_IN_MEMORY:
            logger.info(f"Reached maximum frames in memory ({MAX_FRAMES_IN_MEMORY}), saving checkpoint and clearing memory")
            save_checkpoint(frames, current_second)  # Save one final time before clearing
            frames = []  # Clear frames from memory
            frames, _ = load_checkpoint()  # Reload from checkpoint
            logger.info(f"Reloaded {len(frames)} frames from checkpoint")
                
        current_second += 1
        
    return frames


def extract_one_frame_per_second(video_path, size='default'):
    logger.info(f"Starting frame extraction from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)
    logger.info(f"Video info - FPS: {fps}, Total frames: {total_frames}, Duration: {duration} seconds")
    
    # Get thumbnail dimensions based on size parameter
    width, height = get_thumbnail_dimensions(size)
    
    # Warn about potential memory issues with XL size
    if size == 'xl':
        warn_xl_size(duration)
    
    # Process frames in batches
    frames = process_frames_in_batches(cap, duration, width, height)
    
    cap.release()
    logger.info(f"Completed frame extraction. Total frames: {len(frames)}")
    return frames


def make_composite_thumbnail(frames, thumb_per_row=FRAMES_PER_ROW):
    logger.info(f"Creating composite thumbnail from {len(frames)} frames")
    if not frames:
        logger.error("No frames to create thumbnail")
        return None
    
    w, h = frames[0].size
    rows = math.ceil(len(frames) / thumb_per_row)
    logger.info(f"Creating {rows} rows of thumbnails")
    
    composite = Image.new('RGB', (w * thumb_per_row, h * rows))
    for idx, frame in enumerate(frames):
        x = (idx % thumb_per_row) * w
        y = (idx // thumb_per_row) * h
        composite.paste(Image.fromarray(frame), (x, y))
        if idx % 100 == 0:
            logger.info(f"Processed {idx} frames")
    
    logger.info("Completed composite thumbnail creation")
    return composite


def main():
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Generate a composite video thumbnail.')
        parser.add_argument('video', help='Path to the video file, DVD device, or DVD mount point')
        parser.add_argument('--output', default=OUTPUT_IMAGE, help='Output image file name')
        parser.add_argument('--duration', help='Expected video duration in format HH:MM:SS or MM:SS')
        parser.add_argument('--size', choices=['default', 'xl'], default='default',
                          help='Thumbnail size: default (320x180) or xl (640x360)')
        args = parser.parse_args()

        # Update thumbnail dimensions based on size parameter
        global THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT
        THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT = get_thumbnail_dimensions(args.size)
        logger.info(f"Using thumbnail dimensions: {THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT}")

        logger.info(f"Starting video thumbnail generation for {args.video}")
        video_path = args.video
        temp_file = None
        temp_dir = None

        # Parse expected duration if provided
        expected_duration = None
        if args.duration:
            try:
                expected_duration = parse_duration(args.duration)
                logger.info(f"Expected video duration: {expected_duration} seconds")
            except ValueError as e:
                logger.error(f"Invalid duration format: {str(e)}")
                return

        try:
            if is_dvd_device(video_path):
                logger.info(f"Processing DVD: {video_path}")
                dvd_mount = find_dvd_mount()
                if dvd_mount:
                    video_path = dvd_mount
                temp_file = extract_dvd_to_temp(video_path, expected_duration)
                if not temp_file:
                    logger.error("Failed to extract DVD content")
                    return
                temp_dir = os.path.dirname(temp_file)
                video_path = temp_file

            logger.info(f"Extracting frames from {video_path}")
            frames = extract_one_frame_per_second(video_path, args.size)
            logger.info(f"Extracted {len(frames)} frames")

            logger.info("Creating composite thumbnail")
            composite = make_composite_thumbnail(frames)
            if composite:
                composite.save(args.output)
                logger.info(f"Thumbnail saved as {args.output}")
            else:
                logger.error("No frames extracted. Thumbnail not created.")
        finally:
            # Clean up temporary files
            if temp_file and os.path.exists(temp_file):
                logger.debug(f"Removing temporary file: {temp_file}")
                os.remove(temp_file)
            if temp_dir and os.path.exists(temp_dir):
                try:
                    logger.debug(f"Removing temporary directory: {temp_dir}")
                    os.rmdir(temp_dir)
                except OSError as e:
                    logger.warning(f"Could not remove temporary directory {temp_dir}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main() 
