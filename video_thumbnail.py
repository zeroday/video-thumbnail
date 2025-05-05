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
from typing import List, Tuple, Optional
import shutil
import argparse
import mmap
import array
import psutil
import gc
import json

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
FRAMES_DIR = 'frames'
BATCH_SIZE = 5  # Number of rows to process at once
MEMORY_WARNING_THRESHOLD = 95  # Percentage of total memory
MEMORY_CRITICAL_THRESHOLD = 98  # Percentage of total memory


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
    
    # Check for Chrome OS mount point first
    if path.startswith('/mnt/chromeos/removable/'):
        logger.debug(f"Checking Chrome OS mount point: {path}")
        video_ts_path = os.path.join(path, 'VIDEO_TS')
        if os.path.exists(video_ts_path):
            logger.info(f"Found DVD mount point in Chrome OS: {path}")
            logger.debug(f"VIDEO_TS path exists: {video_ts_path}")
            return True
        else:
            logger.debug(f"VIDEO_TS path does not exist: {video_ts_path}")
    
    # Check for physical DVD devices
    if path.startswith('/dev/dvd') or path.startswith('/dev/sr'):
        logger.info(f"Found DVD device: {path}")
        return True
    
    # Check for VIDEO_TS directory in the path
    video_ts_path = os.path.join(path, 'VIDEO_TS')
    if os.path.exists(video_ts_path):
        logger.info(f"Found DVD mount point: {path}")
        logger.debug(f"VIDEO_TS path exists: {video_ts_path}")
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


def analyze_vob_timeline(vob_files: List[str]) -> List[dict]:
    """Analyze multiple VOB files to create a timeline of their boundaries."""
    logger.info("Analyzing VOB files to determine boundaries")
    vob_info = []
    current_start = 0.0
    
    for vob_file in vob_files:
        try:
            # Get duration using ffprobe
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',  # Only look at video stream
                '-show_entries', 'stream=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                vob_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                # Validate duration (should be between 1 second and 2 hours)
                if 1 <= duration <= 7200:  # 2 hours in seconds
                    vob_info.append({
                        'file': vob_file,
                        'start': current_start,
                        'end': current_start + duration,
                        'duration': duration
                    })
                    current_start += duration
                    logger.info(f"VOB file {vob_file}: {duration:.1f} seconds")
                else:
                    logger.warning(f"Invalid duration {duration:.1f} seconds for {vob_file}, skipping")
            else:
                logger.error(f"Failed to get duration for {vob_file}")
        except Exception as e:
            logger.error(f"Error analyzing {vob_file}: {str(e)}")
    
    if not vob_info:
        logger.error("No valid VOB files found")
        return None
    
    # Sort VOB files by start time
    vob_info.sort(key=lambda x: x['start'])
    return vob_info


def get_vob_file_for_timestamp(vob_info: List[dict], second: float) -> Optional[str]:
    """Find the appropriate VOB file for a given timestamp."""
    for vob in vob_info:
        if vob['start'] <= second < vob['end']:
            return vob['file']
    return None


def extract_frame_from_vob(vob_info: List[dict], second: int, output_file: str, width: int, height: int, enable_gpu: bool = False, max_retries: int = 3) -> bool:
    """Extract a single frame from appropriate VOB file at specified second."""
    try:
        # Find the appropriate VOB file for this second
        vob_file = get_vob_file_for_timestamp(vob_info, second)
        if not vob_file:
            logger.error(f"Could not find VOB file for second {second}")
            return False

        # Base command for VOB extraction
        cmd = [
            'ffmpeg',
            '-ss', str(second),
            '-i', vob_file,
            '-vf', 'yadif=0',  # Deinterlace
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'png',
            '-pix_fmt', 'rgb24',
            '-vsync', '0',
            '-q:v', '2',
            '-threads', '0',
            '-'
        ]

        # Add GPU options if enabled
        if enable_gpu:
            cmd[1:1] = ['-hwaccel', 'cuda', '-hwaccel_device', '0', '-hwaccel_output_format', 'cuda']

        # Run ffmpeg
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()

        if process.returncode != 0:
            logger.error(f"Error extracting frame at {second}s: {err.decode()}")
            return False

        if not out:
            logger.error(f"No image data returned for frame at {second}s")
            return False

        # Convert PNG bytes to numpy array
        nparr = np.frombuffer(out, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error(f"Failed to decode frame at {second}s")
            return False

        # Resize frame
        frame = cv2.resize(frame, (width, height))

        # Save as JPEG with high quality
        cv2.imwrite(output_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Clear memory
        del frame
        del nparr
        del out
        gc.collect()

        return True

    except Exception as e:
        logger.error(f"Exception extracting frame at {second}s: {str(e)}")
        return False


def find_vob_files_in_dvd(dvd_path: str) -> Optional[List[str]]:
    """Find all VOB files in a DVD directory."""
    video_ts_path = os.path.join(dvd_path, 'VIDEO_TS')
    if not os.path.exists(video_ts_path):
        logger.error(f"VIDEO_TS directory not found at {video_ts_path}")
        return None
    
    vob_files = glob.glob(os.path.join(video_ts_path, 'VTS_*.VOB'))
    vob_files.sort()
    
    # Skip menu VOBs
    vob_files = [f for f in vob_files if not f.endswith('_0.VOB')]
    
    if not vob_files:
        logger.error("No VOB files found")
        return None
    
    return vob_files


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
        logger.warning("Processing in very small batches (25 frames) to manage memory usage")
        logger.warning("Maximum frames in memory: 250")
        logger.warning("Memory will be cleared at 200 frames")


def process_frames_in_batches(cap: cv2.VideoCapture, duration: float, width: int, height: int) -> Image.Image:
    """Process frames one at a time, updating the composite image incrementally."""
    # Calculate grid dimensions
    num_rows = int((duration + FRAMES_PER_ROW - 1) // FRAMES_PER_ROW)
    thumbnail_width = int(width * FRAMES_PER_ROW)
    thumbnail_height = int(height * num_rows)
    
    # Create blank thumbnail
    thumbnail = Image.new('RGB', (thumbnail_width, thumbnail_height))
    
    current_second = 0
    frame_count = 0
    
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
        
        # Convert to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Calculate position in grid
        row = frame_count // FRAMES_PER_ROW
        col = frame_count % FRAMES_PER_ROW
        x = col * width
        y = row * height
        
        # Paste frame into composite
        thumbnail.paste(frame_pil, (x, y))
        
        # Clear frame from memory
        del frame
        del frame_pil
        
        # Save checkpoint after each frame
        logger.info(f"Processed frame {frame_count} at {current_second} seconds")
        
        frame_count += 1
        current_second += 1
        
        # Force garbage collection after each frame
        gc.collect()
    
    return thumbnail


def create_thumbnail(frames_or_count, size: str = 'default') -> Image.Image:
    """Create thumbnail image from frames or frame count."""
    if isinstance(frames_or_count, int):
        frame_count = frames_or_count
        if frame_count == 0:
            raise ValueError("No frames to create thumbnail from")
        
        width, height = get_thumbnail_dimensions(size)
        
        # Calculate grid dimensions
        num_rows = (frame_count + FRAMES_PER_ROW - 1) // FRAMES_PER_ROW
        thumbnail_width = width * FRAMES_PER_ROW
        thumbnail_height = height * num_rows
        
        # Create blank thumbnail
        thumbnail = Image.new('RGB', (thumbnail_width, thumbnail_height))
        
        # Load and paste frames one at a time
        for i in range(frame_count):
            frame_path = os.path.join(FRAMES_DIR, f'frame_{i:06d}.jpg')
            if os.path.exists(frame_path):
                frame = Image.open(frame_path)
                row = i // FRAMES_PER_ROW
                col = i % FRAMES_PER_ROW
                x = col * width
                y = row * height
                thumbnail.paste(frame, (x, y))
                frame.close()  # Close the frame file immediately
                
                if i % 100 == 0:
                    logger.info(f"Processed {i} frames")
    else:
        frames = frames_or_count
        if not frames:
            raise ValueError("No frames to create thumbnail from")
        
        width, height = get_thumbnail_dimensions(size)
        
        # Calculate grid dimensions
        num_rows = (len(frames) + FRAMES_PER_ROW - 1) // FRAMES_PER_ROW
        thumbnail_width = width * FRAMES_PER_ROW
        thumbnail_height = height * num_rows
        
        # Create blank thumbnail
        thumbnail = Image.new('RGB', (thumbnail_width, thumbnail_height))
        
        # Paste frames into grid
        for i, frame in enumerate(frames):
            row = i // FRAMES_PER_ROW
            col = i % FRAMES_PER_ROW
            x = col * width
            y = row * height
            
            # Convert BGR to RGB if needed
            if isinstance(frame, np.ndarray):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
            else:
                frame_pil = frame
            
            thumbnail.paste(frame_pil, (x, y))
            
            if i % 100 == 0:
                logger.info(f"Processed {i} frames")
    
    return thumbnail


def cleanup():
    """Clean up temporary files and directories."""
    try:
        if os.path.exists(FRAMES_DIR):
            shutil.rmtree(FRAMES_DIR)
            logger.info(f"Removed frames directory: {FRAMES_DIR}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


def is_vob_file(path):
    """Check if the given path is a VOB file."""
    return path.lower().endswith('.vob')


def get_vob_duration(vob_path):
    """Get duration of VOB file using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',  # Only look at video stream
            '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-i', vob_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            # Also check if this is a video stream
            cmd_info = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_type,width,height',
                '-of', 'json',
                '-i', vob_path
            ]
            info_result = subprocess.run(cmd_info, capture_output=True, text=True)
            if info_result.returncode == 0:
                import json
                info = json.loads(info_result.stdout)
                if 'streams' in info and len(info['streams']) > 0:
                    stream = info['streams'][0]
                    if stream.get('codec_type') == 'video' and stream.get('width', 0) > 0:
                        return duration
            return None
        logger.error(f"ffprobe failed: {result.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error getting VOB duration: {e}")
        return None


def extract_one_frame_per_second(video_path, size='default'):
    """Extract one frame per second from video file."""
    if is_vob_file(video_path):
        return extract_from_vob(video_path, size)
    else:
        return extract_from_mp4(video_path, size)


def get_memory_usage():
    """Get current memory usage percentage."""
    return psutil.virtual_memory().percent


def check_memory_usage():
    """Check memory usage and log warnings if thresholds are exceeded."""
    memory_percent = get_memory_usage()
    if memory_percent >= MEMORY_CRITICAL_THRESHOLD:
        logger.critical(f"Critical memory usage: {memory_percent:.1f}%")
        return True
    elif memory_percent >= MEMORY_WARNING_THRESHOLD:
        logger.warning(f"High memory usage: {memory_percent:.1f}%")
    return False


def process_thumbnail_in_chunks(cap: cv2.VideoCapture, duration: float, width: int, height: int, chunk_size: int = 5) -> Image.Image:
    """Process thumbnail in chunks to reduce memory usage."""
    # Calculate grid dimensions
    num_rows = int((duration + FRAMES_PER_ROW - 1) // FRAMES_PER_ROW)
    thumbnail_width = int(width * FRAMES_PER_ROW)
    thumbnail_height = int(height * num_rows)
    
    # Create chunks directory
    chunks_dir = 'thumbnail_chunks'
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Process in chunks of rows
    for chunk_start in range(0, num_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_rows)
        logger.info(f"Processing rows {chunk_start} to {chunk_end}")
        
        # Check memory usage before starting new chunk
        if check_memory_usage():
            logger.warning("Memory usage high, forcing garbage collection")
            gc.collect()
        
        # Create chunk thumbnail
        chunk_height = height * (chunk_end - chunk_start)
        chunk_thumbnail = Image.new('RGB', (thumbnail_width, chunk_height))
        
        # Process each row in the chunk
        for row in range(chunk_start, chunk_end):
            for col in range(FRAMES_PER_ROW):
                current_second = row * FRAMES_PER_ROW + col
                if current_second >= duration:
                    break
                
                # Check memory usage periodically
                if col % 10 == 0:  # Check every 10 frames
                    if check_memory_usage():
                        logger.warning("Memory usage high, forcing garbage collection")
                        gc.collect()
                
                # Set position to current second
                cap.set(cv2.CAP_PROP_POS_MSEC, current_second * 1000)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame at {current_second} seconds")
                    continue
                
                # Resize frame
                frame = cv2.resize(frame, (width, height))
                
                # Convert to PIL Image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Calculate position in chunk
                x = col * width
                y = (row - chunk_start) * height
                
                # Paste frame into chunk
                chunk_thumbnail.paste(frame_pil, (x, y))
                
                # Clear frame from memory
                del frame
                del frame_pil
                
                # Force garbage collection
                gc.collect()
        
        # Check memory usage before saving chunk
        if check_memory_usage():
            logger.warning("Memory usage high before saving chunk, forcing garbage collection")
            gc.collect()
        
        # Save chunk to disk with compression
        chunk_path = os.path.join(chunks_dir, f'chunk_{chunk_start:04d}.png')
        chunk_thumbnail.save(chunk_path, optimize=True, quality=95)
        logger.info(f"Saved chunk {chunk_start} to {chunk_path}")
        
        # Clear chunk from memory
        del chunk_thumbnail
        gc.collect()
    
    # Combine chunks using memory-mapped files
    final_thumbnail = combine_chunks_memory_mapped(chunks_dir, thumbnail_width, thumbnail_height, chunk_size)
    
    # Clean up chunks directory
    os.rmdir(chunks_dir)
    
    return final_thumbnail


def combine_chunks_memory_mapped(chunks_dir: str, thumbnail_width: int, thumbnail_height: int, chunk_size: int) -> Image.Image:
    """Combine chunks using memory-mapped files to reduce memory usage."""
    logger.info("Combining chunks using memory-mapped files")
    
    # Check initial memory usage
    if check_memory_usage():
        logger.warning("High memory usage before starting combination, forcing garbage collection")
        gc.collect()
    
    # Create a temporary file for the memory mapping
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    
    # Calculate the size needed for the entire image
    image_size = thumbnail_width * thumbnail_height * 3  # 3 bytes per pixel (RGB)
    
    # Create and map the file
    with open(temp_file.name, 'wb') as f:
        # Write zeros to create the file
        f.write(b'\0' * image_size)
    
    # Memory map the file
    with open(temp_file.name, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        
        # Process chunks in smaller groups to reduce memory usage
        chunk_groups = 2  # Reduced from 5 to 2 for even lower memory usage
        num_chunks = (thumbnail_height + chunk_size - 1) // chunk_size
        
        for group_start in range(0, num_chunks, chunk_groups):
            group_end = min(group_start + chunk_groups, num_chunks)
            logger.info(f"Processing chunk group {group_start} to {group_end}")
            
            # Check memory usage before processing group
            if check_memory_usage():
                logger.warning("High memory usage before processing chunk group, forcing garbage collection")
                gc.collect()
            
            # Load and combine chunks in this group
            for chunk_idx in range(group_start, group_end):
                chunk_path = os.path.join(chunks_dir, f'chunk_{chunk_idx:04d}.png')
                if os.path.exists(chunk_path):
                    # Open chunk and process it line by line
                    with Image.open(chunk_path) as chunk:
                        chunk_data = np.array(chunk)
                        
                        # Process the chunk in smaller vertical strips
                        strip_height = 100  # Process 100 pixels at a time
                        for strip_start in range(0, chunk_data.shape[0], strip_height):
                            # Check memory usage periodically
                            if check_memory_usage():
                                logger.warning("High memory usage during strip processing, forcing garbage collection")
                                gc.collect()
                            
                            strip_end = min(strip_start + strip_height, chunk_data.shape[0])
                            strip = chunk_data[strip_start:strip_end]
                            
                            # Calculate the position in the memory map
                            start_pos = (chunk_idx * chunk_size + strip_start) * thumbnail_width * 3
                            
                            # Write strip data to memory map
                            mm.seek(start_pos)
                            mm.write(strip.tobytes())
                            
                            # Clear strip from memory
                            del strip
                            gc.collect()
                    
                    # Clean up
                    os.remove(chunk_path)
                    del chunk_data
                    gc.collect()
        
        # Create final image from memory map in strips
        logger.info("Creating final image from memory map")
        final_thumbnail = Image.new('RGB', (thumbnail_width, thumbnail_height))
        
        # Process the final image in strips
        strip_height = 100
        for strip_start in range(0, thumbnail_height, strip_height):
            # Check memory usage periodically
            if check_memory_usage():
                logger.warning("High memory usage during final image creation, forcing garbage collection")
                gc.collect()
            
            strip_end = min(strip_start + strip_height, thumbnail_height)
            logger.info(f"Processing final image strip {strip_start} to {strip_end}")
            
            # Read strip from memory map
            mm.seek(strip_start * thumbnail_width * 3)
            strip_data = mm.read((strip_end - strip_start) * thumbnail_width * 3)
            strip_array = np.frombuffer(strip_data, dtype=np.uint8)
            strip_array = strip_array.reshape((strip_end - strip_start, thumbnail_width, 3))
            
            # Convert strip to PIL Image and paste into final image
            strip_image = Image.fromarray(strip_array)
            final_thumbnail.paste(strip_image, (0, strip_start))
            
            # Clean up
            del strip_data
            del strip_array
            del strip_image
            gc.collect()
        
        # Clean up
        mm.close()
    
    # Remove temporary file
    os.unlink(temp_file.name)
    
    return final_thumbnail


def extract_from_mp4(video_path, size):
    """Extract frames from MP4 file using OpenCV."""
    logger.info(f"Starting MP4 extraction from {video_path}")
    
    # Get video duration
    duration = get_video_duration(video_path)
    if duration is None:
        raise ValueError("Could not determine video duration")
    
    # Process frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Calculate dimensions
    width, height = get_thumbnail_dimensions(size)
    
    try:
        # Process in chunks
        thumbnail = process_thumbnail_in_chunks(cap, duration, width, height)
        return thumbnail
    finally:
        cap.release()


def build_vob_timeline(vob_files):
    """Build a timeline mapping from global seconds to (vob_file, local_offset)."""
    timeline = []
    current_start = 0.0
    for vob_file in vob_files:
        duration = get_vob_duration(vob_file)
        if duration is None:
            logger.warning(f"Could not determine duration for {vob_file}, skipping.")
            continue
        timeline.append({
            'vob_file': vob_file,
            'start': current_start,
            'end': current_start + duration,
            'duration': duration
        })
        current_start += duration
    return timeline


def find_vob_for_time(timeline, global_second):
    """Find the VOB file and local offset for a given global second."""
    for entry in timeline:
        if entry['start'] <= global_second < entry['end']:
            # Clamp local_offset to just before the end of the VOB
            local_offset = min(global_second - entry['start'], entry['duration'] - 0.5)
            return entry['vob_file'], local_offset
    # If not found, return the last VOB file and its last valid frame
    if timeline:
        entry = timeline[-1]
        return entry['vob_file'], max(0, entry['duration'] - 0.5)
    return None, None


def extract_from_vob(vob_path, size, enable_gpu=False):
    """Extract frames directly from VOB file with correct mapping for global time."""
    logger.info(f"Starting direct VOB extraction from {vob_path}")
    
    # Get video duration from the main movie info
    video_ts_path = os.path.dirname(vob_path)
    movie_info = analyze_vob_timeline(video_ts_path)
    if not movie_info:
        logger.error("Failed to get movie information")
        return None
    
    duration = int(movie_info['duration'])
    logger.info(f"Video duration: {duration:.2f} seconds")
    vob_files = movie_info['vob_files']
    
    # Build timeline mapping
    timeline = build_vob_timeline(vob_files)
    if not timeline:
        logger.error("Could not build VOB timeline.")
        return None
    
    # Calculate dimensions
    width, height = get_thumbnail_dimensions(size)
    num_rows = int((duration + FRAMES_PER_ROW - 1) // FRAMES_PER_ROW)
    thumbnail_width = width * FRAMES_PER_ROW
    thumbnail_height = height * num_rows
    
    # Create chunks directory
    chunks_dir = 'thumbnail_chunks'
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Process in chunks of rows
    chunk_size = 5  # Reduced from 10 to 5 for even lower memory usage
    for chunk_start in range(0, num_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_rows)
        logger.info(f"Processing rows {chunk_start} to {chunk_end}")
        
        # Create chunk thumbnail
        chunk_height = height * (chunk_end - chunk_start)
        chunk_thumbnail = Image.new('RGB', (thumbnail_width, chunk_height))
        
        # Process each row in the chunk
        for row in range(chunk_start, chunk_end):
            for col in range(FRAMES_PER_ROW):
                global_second = row * FRAMES_PER_ROW + col
                if global_second >= duration:
                    break
                
                vob_file, local_offset = find_vob_for_time(timeline, global_second)
                if vob_file is None or local_offset < 0 or local_offset >= get_vob_duration(vob_file):
                    logger.warning(f'Skipping frame at {global_second}s: out of bounds for {vob_file}')
                    continue
                
                frame = None
                try:
                    # Use ffmpeg to extract frame at specific local timestamp
                    cmd = [
                        'ffmpeg',
                        '-ss', str(local_offset),
                        '-i', vob_file,
                        '-vf', 'yadif=0',  # Deinterlace
                        '-vframes', '1',
                        '-f', 'image2pipe',
                        '-vcodec', 'png',
                        '-pix_fmt', 'rgb24',
                        '-vsync', '0',
                        '-q:v', '2',
                        '-threads', '0',
                        '-'
                    ]
                    # Add GPU options if enabled
                    gpu_options = get_gpu_options(enable_gpu)
                    if gpu_options:
                        cmd[1:1] = gpu_options
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = process.communicate()
                    if process.returncode == 0 and out:
                        nparr = np.frombuffer(out, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = cv2.resize(frame, (width, height))
                except Exception as e:
                    logger.error(f"Exception extracting frame: global_second={global_second}, vob_file={vob_file}, local_offset={local_offset}, cmd={' '.join(cmd)}, error={e}")
                
                if frame is not None:
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    x = col * width
                    y = (row - chunk_start) * height
                    chunk_thumbnail.paste(frame_pil, (x, y))
                    del frame
                    del frame_pil
                else:
                    logger.error(f"Failed to extract frame: global_second={global_second}, vob_file={vob_file}, local_offset={local_offset}, cmd={' '.join(cmd)}, returncode={process.returncode if 'process' in locals() else 'N/A'}, stderr={err.decode() if 'err' in locals() else 'N/A'}")
                    black_frame = Image.new('RGB', (width, height))
                    x = col * width
                    y = (row - chunk_start) * height
                    chunk_thumbnail.paste(black_frame, (x, y))
                import gc
                gc.collect()
        chunk_path = os.path.join(chunks_dir, f'chunk_{chunk_start:04d}.png')
        chunk_thumbnail.save(chunk_path, optimize=True, quality=95)
        logger.info(f"Saved chunk {chunk_start} to {chunk_path}")
        del chunk_thumbnail
        gc.collect()
    final_thumbnail = combine_chunks_memory_mapped(chunks_dir, thumbnail_width, thumbnail_height, chunk_size)
    os.rmdir(chunks_dir)
    return final_thumbnail


def analyze_ifo_file(ifo_path: str) -> Optional[dict]:
    """Analyze IFO file to get video information."""
    try:
        # First try to get title information
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-i', ifo_path,
            '-show_entries', 'format=duration',
            '-show_entries', 'stream=codec_type,duration,width,height',
            '-of', 'json'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            logger.debug(f"IFO analysis for {ifo_path}:")
            logger.debug(json.dumps(info, indent=2))
            
            # Get format duration
            format_duration = float(info.get('format', {}).get('duration', 0))
            
            # Get video stream info
            video_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                return {
                    'format_duration': format_duration,
                    'stream_duration': float(video_stream.get('duration', 0)),
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0))
                }
            elif format_duration > 0:
                return {
                    'format_duration': format_duration,
                    'stream_duration': 0,
                    'width': 0,
                    'height': 0
                }
        
        # If ffprobe fails, try using mplayer to get info
        cmd = [
            'mplayer',
            '-identify',
            '-frames', '0',
            '-vo', 'null',
            '-ao', 'null',
            ifo_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse mplayer output
            duration = 0
            width = 0
            height = 0
            for line in result.stdout.split('\n'):
                if 'ID_LENGTH=' in line:
                    duration = float(line.split('=')[1])
                elif 'ID_VIDEO_WIDTH=' in line:
                    width = int(line.split('=')[1])
                elif 'ID_VIDEO_HEIGHT=' in line:
                    height = int(line.split('=')[1])
            
            if duration > 0:
                return {
                    'format_duration': duration,
                    'stream_duration': duration,
                    'width': width,
                    'height': height
                }
        
        logger.warning(f"Could not get valid information from {ifo_path}")
        return None
    except Exception as e:
        logger.error(f"Error analyzing IFO file {ifo_path}: {str(e)}")
        return None


def analyze_vob_file(vob_path: str) -> Optional[dict]:
    """Analyze a single VOB file to get its duration and frame information."""
    try:
        # First try to get basic info
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,r_frame_rate,width,height',
            '-of', 'json',
            vob_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            if 'streams' in info and len(info['streams']) > 0:
                stream = info['streams'][0]
                duration = float(stream.get('duration', 0))
                
                # Get file size
                file_size = os.path.getsize(vob_path)
                file_size_mb = file_size / (1024 * 1024)
                
                # Log detailed information about the VOB file
                logger.info(f"\nAnalyzing VOB file: {os.path.basename(vob_path)}")
                logger.info(f"  File size: {file_size_mb:.1f}MB")
                logger.info(f"  Raw duration: {duration:.1f} seconds")
                
                # Skip files smaller than 1MB
                if file_size < 1000000:  # 1MB
                    logger.warning(f"  Skipping: File too small ({file_size_mb:.1f}MB)")
                    return None
                
                # For very long durations, try to get more accurate duration
                if duration > 7200:  # More than 2 hours
                    logger.warning(f"  Warning: Unusually long duration ({duration/3600:.1f} hours)")
                    # Try to get duration from format instead of stream
                    cmd_format = [
                        'ffprobe',
                        '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'json',
                        vob_path
                    ]
                    format_result = subprocess.run(cmd_format, capture_output=True, text=True)
                    if format_result.returncode == 0:
                        format_info = json.loads(format_result.stdout)
                        format_duration = float(format_info.get('format', {}).get('duration', 0))
                        if 0 < format_duration < 7200:  # If format duration is more reasonable
                            logger.info(f"  Using format duration instead: {format_duration:.1f} seconds")
                            duration = format_duration
                
                # Validate duration (should be between 1 second and 2 hours)
                if duration < 1 or duration > 7200:  # 2 hours in seconds
                    logger.warning(f"  Skipping: Invalid duration ({duration:.1f} seconds)")
                    return None
                
                # Parse frame rate (e.g., "30000/1001" -> 29.97)
                frame_rate = stream.get('r_frame_rate', '0/1')
                if '/' in frame_rate:
                    num, den = map(int, frame_rate.split('/'))
                    fps = num / den if den != 0 else 0
                else:
                    fps = float(frame_rate)
                
                # Validate frame rate (should be between 23.976 and 60)
                if fps < 23.976 or fps > 60:
                    logger.warning(f"  Skipping: Invalid frame rate ({fps:.3f})")
                    return None
                
                logger.info(f"  Valid duration: {duration:.1f} seconds")
                logger.info(f"  Frame rate: {fps:.3f} fps")
                logger.info(f"  Resolution: {stream.get('width', 0)}x{stream.get('height', 0)}")
                
                return {
                    'duration': duration,
                    'fps': fps,
                    'width': int(stream.get('width', 0)),
                    'height': int(stream.get('height', 0)),
                    'total_frames': int(duration * fps),
                    'file_size': file_size
                }
        
        # If ffprobe fails, try mplayer as fallback
        cmd = [
            'mplayer',
            '-identify',
            '-frames', '0',
            '-vo', 'null',
            '-ao', 'null',
            vob_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = 0
            fps = 0
            width = 0
            height = 0
            for line in result.stdout.split('\n'):
                if 'ID_LENGTH=' in line:
                    duration = float(line.split('=')[1])
                elif 'ID_VIDEO_FPS=' in line:
                    fps = float(line.split('=')[1])
                elif 'ID_VIDEO_WIDTH=' in line:
                    width = int(line.split('=')[1])
                elif 'ID_VIDEO_HEIGHT=' in line:
                    height = int(line.split('=')[1])
            
            if duration > 0 and fps > 0:
                # Apply same validations as above
                if duration < 1 or duration > 7200:
                    logger.warning(f"  Skipping: Invalid duration ({duration:.1f} seconds)")
                    return None
                if fps < 23.976 or fps > 60:
                    logger.warning(f"  Skipping: Invalid frame rate ({fps:.3f})")
                    return None
                
                file_size = os.path.getsize(vob_path)
                if file_size < 1000000:
                    logger.warning(f"  Skipping: File too small ({file_size/1024/1024:.1f}MB)")
                    return None
                
                logger.info(f"  Valid duration (from mplayer): {duration:.1f} seconds")
                logger.info(f"  Frame rate: {fps:.3f} fps")
                logger.info(f"  Resolution: {width}x{height}")
                
                return {
                    'duration': duration,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': int(duration * fps),
                    'file_size': file_size
                }
        
        logger.warning(f"  Skipping: Could not get valid information")
        return None
    except Exception as e:
        logger.error(f"Error analyzing VOB file {vob_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def create_extraction_plan(dvd_path: str, target_duration: int) -> Optional[List[dict]]:
    """Create a plan for VOB file extraction based on IFO analysis."""
    logger.info("Starting extraction planning phase")
    
    # Check for VIDEO_TS directory
    video_ts_path = os.path.join(dvd_path, 'VIDEO_TS')
    if not os.path.exists(video_ts_path):
        logger.error(f"VIDEO_TS directory not found at {video_ts_path}")
        return None
    
    # Find all IFO files
    ifo_files = glob.glob(os.path.join(video_ts_path, 'VTS_*.IFO'))
    if not ifo_files:
        logger.error("No IFO files found")
        return None
    
    # Sort IFO files
    ifo_files.sort()
    logger.info(f"Found {len(ifo_files)} IFO files")
    
    # First pass: analyze all titles
    all_titles = []
    
    for ifo_file in ifo_files:
        logger.info(f"Analyzing IFO file: {ifo_file}")
        info = analyze_ifo_file(ifo_file)
        if not info:
            logger.warning(f"No valid information found in {ifo_file}")
            continue
            
        # Get corresponding VOB files
        vts_num = os.path.basename(ifo_file).split('_')[1]
        vob_files = glob.glob(os.path.join(video_ts_path, f'VTS_{vts_num}_*.VOB'))
        vob_files.sort()
        
        # Skip menu VOBs
        vob_files = [f for f in vob_files if not f.endswith('_0.VOB')]
        
        if not vob_files:
            logger.warning(f"No VOB files found for {ifo_file}")
            continue
            
        # Use the longer duration between format and stream
        title_duration = max(info['format_duration'], info['stream_duration'])
        
        # Skip very short titles (likely menus or extras)
        if title_duration < 60:  # Skip titles shorter than 1 minute
            logger.info(f"Skipping short title: {title_duration/60:.1f} minutes")
            continue
            
        # Skip titles with unusual resolutions
        if info['width'] < 640 or info['height'] < 480:
            logger.info(f"Skipping low resolution title: {info['width']}x{info['height']}")
            continue
        
        # Add to all titles list
        all_titles.append({
            'ifo_file': ifo_file,
            'vob_files': vob_files,
            'duration': title_duration,
            'width': info['width'],
            'height': info['height'],
            'vts_num': vts_num
        })
        logger.info(f"Found title {vts_num}: {title_duration/60:.1f} minutes, {info['width']}x{info['height']}")
    
    if not all_titles:
        logger.error("No valid titles found in IFO files")
        return None
    
    # Sort titles by duration (longest first)
    all_titles.sort(key=lambda x: x['duration'], reverse=True)
    
    # Select the longest title that's close to the target duration
    main_title = None
    for title in all_titles:
        # Check if duration is within 10% of target
        if abs(title['duration'] - target_duration) / target_duration <= 0.1:
            main_title = title
            break
    
    # If no title matches target duration, use the longest one
    if not main_title:
        main_title = all_titles[0]
        logger.warning(f"No title matches target duration of {target_duration/60:.1f} minutes")
        logger.warning(f"Using longest title: {main_title['duration']/60:.1f} minutes")
    
    # Analyze each VOB file in the main title
    logger.info("\nAnalyzing VOB files for main title:")
    current_second = 0
    vob_details = []
    total_vob_duration = 0
    
    # Calculate expected duration per VOB file
    num_vob_files = len(main_title['vob_files'])
    expected_duration_per_vob = main_title['duration'] / num_vob_files
    
    for vob_file in main_title['vob_files']:
        vob_info = analyze_vob_file(vob_file)
        if vob_info:
            # If the duration is way off from expected, use the expected duration
            if abs(vob_info['duration'] - expected_duration_per_vob) > expected_duration_per_vob * 0.5:
                logger.warning(f"  Correcting duration for {os.path.basename(vob_file)}")
                logger.warning(f"    Reported: {vob_info['duration']:.1f} seconds")
                logger.warning(f"    Expected: {expected_duration_per_vob:.1f} seconds")
                vob_info['duration'] = expected_duration_per_vob
            
            start_frame = current_second
            end_frame = current_second + int(vob_info['duration'])
            vob_details.append({
                'file': vob_file,  # Use full path
                'duration': vob_info['duration'],
                'fps': vob_info['fps'],
                'start': start_frame,  # Changed from start_frame to start
                'end': end_frame,      # Changed from end_frame to end
                'frame_range': f"frame-{start_frame:04d}.jpg to frame-{end_frame:04d}.jpg"
            })
            current_second = end_frame
            total_vob_duration += vob_info['duration']
    
    # Create plan with the selected title
    vob_plan = [{
        'ifo_file': main_title['ifo_file'],
        'vob_files': main_title['vob_files'],
        'duration': main_title['duration'],
        'start_time': 0,  # Start from beginning
        'width': main_title['width'],
        'height': main_title['height'],
        'vob_details': vob_details
    }]
    
    # Print plan
    logger.info("\nExtraction Plan:")
    logger.info(f"1. Target duration: {target_duration/60:.1f} minutes ({target_duration} seconds)")
    logger.info(f"2. IFO duration: {main_title['duration']/60:.1f} minutes ({main_title['duration']} seconds)")
    logger.info(f"3. Total VOB duration: {total_vob_duration/60:.1f} minutes ({total_vob_duration} seconds)")
    
    # Validate duration against acceptable range (±30 minutes)
    # Use IFO duration as the reference point since it's more accurate
    duration_diff = abs(main_title['duration'] - target_duration)
    if duration_diff > 1800:  # 30 minutes in seconds
        logger.error(f"ERROR: IFO duration ({main_title['duration']/60:.1f} minutes) is outside acceptable range")
        logger.error(f"Difference: {duration_diff/60:.1f} minutes ({duration_diff} seconds)")
        logger.error("Acceptable range: ±30 minutes from target duration")
        return None
    else:
        logger.info(f"4. Duration difference: {duration_diff/60:.1f} minutes ({duration_diff} seconds)")
        logger.info("   ✓ Within acceptable range (±30 minutes)")
    
    logger.info(f"\nResolution: {main_title['width']}x{main_title['height']}")
    logger.info("\nVOB files and frame ranges:")
    
    # Print each VOB file's details
    for detail in vob_details:
        logger.info(f"\n  {os.path.basename(detail['file'])}:")
        logger.info(f"    Duration: {detail['duration']/60:.1f} minutes ({detail['duration']:.1f} seconds)")
        logger.info(f"    Frame rate: {detail['fps']:.2f} fps")
        logger.info(f"    Frame range: {detail['frame_range']}")
    
    # Print summary of frame ranges
    logger.info("\nFrame range summary:")
    if vob_details:
        first_frame = vob_details[0]['start']
        last_frame = vob_details[-1]['end']
        logger.info(f"  First frame: frame-{first_frame:04d}.jpg")
        logger.info(f"  Last frame: frame-{last_frame:04d}.jpg")
        logger.info(f"  Total frames: {last_frame - first_frame}")
    
    # Print other titles for reference
    logger.info("\nOther titles found:")
    for title in all_titles:
        if title != main_title:
            logger.info(f"  Title {title['vts_num']}: {title['duration']/60:.1f} minutes, {title['width']}x{title['height']}")
    
    return vob_plan


def is_warning_screen(frame: np.ndarray, blue_threshold: float = 0.7, motion_threshold: float = 0.1) -> bool:
    """Detect if a frame is a warning screen (mostly blue with little motion)."""
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define blue color range in HSV
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        
        # Create mask for blue pixels
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = cv2.countNonZero(blue_mask) / (frame.shape[0] * frame.shape[1])
        
        # Check if frame is mostly blue
        return blue_ratio > blue_threshold
    except Exception as e:
        logger.error(f"Error in is_warning_screen: {str(e)}")
        return False


def find_movie_start(vob_info: List[dict], width: int, height: int, enable_gpu: bool = False, max_seconds: int = 120) -> int:
    """Find the start of the actual movie by analyzing the first few minutes."""
    logger.info("Analyzing first few minutes to find movie start...")
    
    # Create temporary directory for analysis frames
    temp_dir = 'temp_analysis'
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        last_frame = None
        movie_start = 0
        
        # Analyze frames up to max_seconds
        for second in range(max_seconds):
            temp_file = os.path.join(temp_dir, f'temp_{second:04d}.jpg')
            
            # Extract frame
            if not extract_frame_from_vob(vob_info, second, temp_file, width, height, enable_gpu):
                continue
            
            # Read frame
            frame = cv2.imread(temp_file)
            if frame is None:
                continue
            
            # Check if this is a warning screen
            if is_warning_screen(frame):
                logger.debug(f"Warning screen detected at {second}s")
                continue
            
            # If we get here, we found a non-warning frame
            movie_start = second
            logger.info(f"Found movie start at {second}s")
            break
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return movie_start


def extract_all_frames(dvd_path: str, duration: int, size: str, enable_gpu: bool = False) -> bool:
    """Extract one frame per second from DVD."""
    try:
        # Create extraction plan
        vob_plan = create_extraction_plan(dvd_path, duration)
        if not vob_plan:
            logger.error("Failed to create extraction plan")
            return False

        # Get thumbnail dimensions
        width, height = get_thumbnail_dimensions(size)
        
        # Warn about XL size for long videos
        warn_xl_size(duration)

        # Create frames directory if it doesn't exist
        frames_dir = 'frames'
        os.makedirs(frames_dir, exist_ok=True)

        # Get VOB details from plan
        vob_details = vob_plan[0]['vob_details']
        
        # Find movie start
        movie_start = find_movie_start(vob_details, width, height, enable_gpu)
        if movie_start > 0:
            logger.info(f"Found movie start at {movie_start} seconds")
        else:
            logger.info("No warning screens detected, starting from beginning")
            movie_start = 0

        # Find the last extracted frame
        existing_frames = glob.glob(os.path.join(frames_dir, 'frame-*.jpg'))
        if existing_frames:
            # Extract frame numbers and find the highest
            frame_numbers = [int(os.path.basename(f).split('-')[1].split('.')[0]) for f in existing_frames]
            frames_extracted = max(frame_numbers) + 1
            logger.info(f"Resuming from frame {frames_extracted}")
        else:
            frames_extracted = 0

        # Extract frames
        total_frames = duration
        logger.info(f"\nExtracting {total_frames} frames...")
        
        consecutive_failures = 0
        max_consecutive_failures = 10  # Allow up to 10 consecutive failures before giving up
        
        for second in range(movie_start + frames_extracted, movie_start + duration):
            # Check memory usage every 100 frames
            if frames_extracted % 100 == 0:
                check_memory_usage()
            
            output_file = os.path.join(frames_dir, f'frame-{frames_extracted:04d}.jpg')
            
            # Skip if frame already exists
            if os.path.exists(output_file):
                frames_extracted += 1
                consecutive_failures = 0  # Reset failure counter on successful skip
                if frames_extracted % 60 == 0:  # Log progress every minute
                    logger.info(f"Extracted {frames_extracted}/{total_frames} frames")
                continue
            
            if extract_frame_from_vob(vob_details, second, output_file, width, height, enable_gpu):
                frames_extracted += 1
                consecutive_failures = 0  # Reset failure counter on success
                if frames_extracted % 60 == 0:  # Log progress every minute
                    logger.info(f"Extracted {frames_extracted}/{total_frames} frames")
            else:
                consecutive_failures += 1
                logger.warning(f"Failed to extract frame at second {second} (consecutive failures: {consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping extraction")
                    break
                
                # Try to extract the next frame
                continue

        if frames_extracted > 0:
            logger.info(f"Successfully extracted {frames_extracted} frames")
            return True
        else:
            logger.error("No frames were successfully extracted")
            return False

    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate a composite video thumbnail.')
    parser.add_argument('video', help='Path to the DVD mount point')
    parser.add_argument('--output', default=OUTPUT_IMAGE, help='Output image file name')
    parser.add_argument('--duration', required=True, help='Expected video duration in format HH:MM:SS or MM:SS')
    parser.add_argument('--size', choices=['default', 'xl'], default='default',
                      help='Thumbnail size: default (320x180) or xl (640x360)')
    parser.add_argument('--enable-gpu', action='store_true',
                      help='Enable GPU acceleration if available')
    parser.add_argument('--skip-warning', action='store_true',
                      help='Skip warning screens at the start of the movie')
    args = parser.parse_args()
    
    try:
        # Parse duration
        duration_parts = args.duration.split(':')
        if len(duration_parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, duration_parts)
            duration = hours * 3600 + minutes * 60 + seconds
        elif len(duration_parts) == 2:  # MM:SS
            minutes, seconds = map(int, duration_parts)
            duration = minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid duration format: {args.duration}")
        
        logger.info(f"Processing video with duration: {duration} seconds")
        logger.info(f"Will create {duration} frames (one per second)")
        
        # Create extraction plan
        vob_plan = create_extraction_plan(args.video, duration)
        if not vob_plan:
            raise Exception("Failed to create extraction plan")
        
        # Ask for confirmation
        print("\nDo you want to proceed with this extraction plan? [y/n]")
        response = input().lower()
        if response != 'y':
            logger.info("Extraction cancelled by user")
            return
        
        # Extract frames
        if not extract_all_frames(args.video, duration, args.size, args.enable_gpu):
            raise Exception("Frame extraction failed")
        
        # Assemble final image
        thumbnail = create_thumbnail(duration, args.size)
        if thumbnail is None:
            raise Exception("Failed to assemble final image")

        # Save the final thumbnail
        thumbnail.save(args.output)
        logger.info(f"Thumbnail saved to {args.output}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        cleanup()


if __name__ == "__main__":
    main() 
