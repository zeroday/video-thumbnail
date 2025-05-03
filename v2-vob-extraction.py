#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
from PIL import Image
import subprocess
import logging
import argparse
from pathlib import Path
import tempfile
import shutil
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 300  # Process frames in batches to manage memory
CHECKPOINT_FILE = 'thumbnail_checkpoint.npz'

def get_thumbnail_dimensions(size: str) -> Tuple[int, int]:
    """Get thumbnail dimensions based on size parameter."""
    if size == 'xl':
        return (640, 360)  # 16:9 aspect ratio
    return (320, 180)  # Default size, 16:9 aspect ratio

def get_vob_duration(vob_path: str) -> float:
    """Get duration of VOB file using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            vob_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting VOB duration: {e}")
        raise

def extract_frames_from_vob(vob_path: str, positions: List[float], size: str) -> List[np.ndarray]:
    """Extract frames from VOB file at specified positions."""
    width, height = get_thumbnail_dimensions(size)
    frames = []
    
    for position in positions:
        try:
            # Use ffmpeg to extract frame at specific timestamp
            cmd = [
                'ffmpeg',
                '-ss', str(position),
                '-i', vob_path,
                '-vframes', '1',
                '-f', 'image2pipe',
                '-vcodec', 'png',
                '-'
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error extracting frame at {position}s: {err.decode()}")
                continue
                
            # Convert PNG bytes to numpy array
            nparr = np.frombuffer(out, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Resize frame to thumbnail dimensions
                frame = cv2.resize(frame, (width, height))
                frames.append(frame)
                
        except Exception as e:
            logger.error(f"Error processing frame at {position}s: {e}")
            continue
            
    return frames

def extract_one_frame_per_second(vob_path: str, size: str = 'default') -> List[np.ndarray]:
    """Extract one frame per second from VOB file."""
    logger.info(f"Starting frame extraction from {vob_path}")
    
    # Get video duration
    duration = get_vob_duration(vob_path)
    logger.info(f"Video duration: {duration:.2f} seconds")
    
    # Calculate frame positions (one per second)
    frame_positions = list(range(int(duration)))
    logger.info(f"Will extract {len(frame_positions)} frames")
    
    # Process frames in batches
    all_frames = []
    for i in range(0, len(frame_positions), BATCH_SIZE):
        batch_positions = frame_positions[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {(len(frame_positions) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        batch_frames = extract_frames_from_vob(vob_path, batch_positions, size)
        all_frames.extend(batch_frames)
        
        # Save checkpoint after each batch
        save_checkpoint(all_frames, len(all_frames))
    
    logger.info(f"Successfully extracted {len(all_frames)} frames")
    return all_frames

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

def create_thumbnail(frames: List[np.ndarray], size: str = 'default') -> Image.Image:
    """Create thumbnail image from frames."""
    if not frames:
        raise ValueError("No frames to create thumbnail from")
    
    width, height = get_thumbnail_dimensions(size)
    frames_per_row = 60  # One minute per row
    
    # Calculate grid dimensions
    num_rows = (len(frames) + frames_per_row - 1) // frames_per_row
    thumbnail_width = width * frames_per_row
    thumbnail_height = height * num_rows
    
    # Create blank thumbnail
    thumbnail = Image.new('RGB', (thumbnail_width, thumbnail_height))
    
    # Paste frames into grid
    for i, frame in enumerate(frames):
        row = i // frames_per_row
        col = i % frames_per_row
        x = col * width
        y = row * height
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        thumbnail.paste(frame_pil, (x, y))
    
    return thumbnail

def warn_xl_size(duration: float):
    """Warn user about potential memory issues with XL size."""
    if duration > 3600:  # More than 1 hour
        logger.warning("Using XL size for videos longer than 1 hour may cause memory issues.")
        logger.warning("Consider using default size for very long videos.")

def main():
    parser = argparse.ArgumentParser(description='Generate thumbnail from VOB file')
    parser.add_argument('vob_path', help='Path to VOB file')
    parser.add_argument('--size', choices=['default', 'xl'], default='default',
                      help='Thumbnail size (default: 320x180, xl: 640x360)')
    parser.add_argument('--output', default='video_thumbnail.jpg',
                      help='Output image file name')
    args = parser.parse_args()
    
    try:
        # Check if VOB file exists
        if not os.path.exists(args.vob_path):
            raise FileNotFoundError(f"VOB file not found: {args.vob_path}")
        
        # Get video duration and warn about XL size if needed
        duration = get_vob_duration(args.vob_path)
        warn_xl_size(duration)
        
        # Try to load checkpoint
        frames, processed_count = load_checkpoint()
        
        if not frames:
            # Extract frames if no checkpoint found
            frames = extract_one_frame_per_second(args.vob_path, args.size)
        else:
            logger.info(f"Resuming from checkpoint with {processed_count} frames")
            remaining_frames = extract_one_frame_per_second(args.vob_path, args.size)
            frames.extend(remaining_frames)
        
        # Create and save thumbnail
        thumbnail = create_thumbnail(frames, args.size)
        thumbnail.save(args.output)
        logger.info(f"Thumbnail saved to {args.output}")
        
        # Clean up checkpoint file
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info("Checkpoint file cleaned up")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 