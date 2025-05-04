# Video Thumbnail Generator

A Python script that generates a composite thumbnail image from video files, creating a visual timeline of the content. The script can process both standard video files and DVD VOB files.

## Features

- Generate composite thumbnails from video files
- Memory-efficient frame processing
- Support for different thumbnail sizes (320x180 or 640x360)
- Checkpoint functionality for resuming interrupted processes
- Detailed logging
- Automatic cleanup of temporary files
- DVD VOB file support (secondary feature)

## Requirements

- Python 3.6 or higher
- OpenCV (`opencv-python`)
- Pillow (`Pillow`)
- NumPy (`numpy`)
- FFmpeg and FFprobe

## Installation

1. Install Python dependencies:
```bash
pip install opencv-python Pillow numpy
```

2. Install FFmpeg and FFprobe:
```bash
# On Ubuntu/Debian
sudo apt-get install ffmpeg

# On macOS
brew install ffmpeg

# On Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

### Basic Video Processing
```bash
python video_thumbnail.py path/to/video.mp4
```

### Advanced Options
```bash
# Specify output file
python video_thumbnail.py path/to/video.mp4 --output my_thumbnail.jpg

# Use XL size (640x360)
python video_thumbnail.py path/to/video.mp4 --size xl
```

### DVD VOB Processing
```bash
# Basic VOB processing
python video_thumbnail.py path/to/video.vob

# VOB processing with duration
python video_thumbnail.py path/to/video.vob --duration 02:30:00
```

## How It Works

1. The script extracts one frame per second from the video
2. Frames are arranged in a grid:
   - Each row contains 60 frames (one per second, representing one minute)
   - Each row represents one minute of video
   - Multiple rows represent multiple minutes
3. The final composite image provides a visual timeline of the video content

## Testing

### Test Cases

1. Short video (1-5 minutes):
```bash
python video_thumbnail.py test/short.mp4
```

2. Medium video (30-60 minutes):
```bash
python video_thumbnail.py test/medium.mp4 --size xl
```

3. Long video (1+ hours):
```bash
python video_thumbnail.py test/long.mp4
```

### Expected Results

- Short videos: Quick processing, clear thumbnails
- Medium videos: Moderate processing time, good quality with XL size
- Long videos: Longer processing time, memory warnings for XL size

### Testing Checkpoint Functionality

1. Start processing a long video:
```bash
python video_thumbnail.py test/long.mp4
```

2. Interrupt the process (Ctrl+C)

3. Resume processing:
```bash
python video_thumbnail.py test/long.mp4
```

The script should resume from the last processed frame.

## Troubleshooting

1. "Video file not found":
   - Ensure the file path is correct
   - Check file permissions

2. Memory issues with large videos:
   - Use default size (320x180) instead of XL
   - Process in smaller segments

3. VOB processing issues:
   - Ensure the DVD is properly mounted
   - Provide the correct duration with --duration parameter
   - Check VOB file permissions

## Notes

- The script maintains 16:9 aspect ratio for all thumbnail sizes
- For DVD processing, always provide the --duration parameter
- XL size (640x360) requires more memory and processing time
- Temporary files are automatically cleaned up after processing 