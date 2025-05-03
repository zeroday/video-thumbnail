# VOB Thumbnail Generator

A Python script that generates a composite thumbnail image directly from VOB files without intermediate conversion.

## Features

- Direct extraction of frames from VOB files
- Memory-efficient batch processing
- Checkpoint functionality for resuming interrupted processes
- Support for different thumbnail sizes
- Detailed logging
- Automatic cleanup of temporary files

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

Basic usage:
```bash
python v2-vob-extraction.py path/to/video.vob
```

Advanced options:
```bash
# Specify output file
python v2-vob-extraction.py path/to/video.vob --output my_thumbnail.jpg

# Use XL size (640x360)
python v2-vob-extraction.py path/to/video.vob --size xl
```

## Testing

### Test Cases

1. Short video (1-5 minutes):
```bash
python v2-vob-extraction.py test/short.vob
```

2. Medium video (30-60 minutes):
```bash
python v2-vob-extraction.py test/medium.vob --size xl
```

3. Long video (1+ hours):
```bash
python v2-vob-extraction.py test/long.vob
```

### Expected Results

- Short videos: Quick processing, clear thumbnails
- Medium videos: Moderate processing time, good quality with XL size
- Long videos: Longer processing time, memory warnings for XL size

### Testing Checkpoint Functionality

1. Start processing a long video:
```bash
python v2-vob-extraction.py test/long.vob
```

2. Interrupt the process (Ctrl+C)

3. Resume processing:
```bash
python v2-vob-extraction.py test/long.vob
```

The script should resume from the last processed frame.

## Troubleshooting

1. "VOB file not found":
   - Verify the file path
   - Check file permissions

2. "Error getting VOB duration":
   - Ensure FFprobe is installed
   - Check if the VOB file is valid

3. Memory issues with XL size:
   - Use default size for very long videos
   - Increase system swap space if needed

## Notes

- The script processes frames in batches of 300 to manage memory usage
- Checkpoint files are automatically cleaned up after successful completion
- XL size (640x360) is recommended for videos under 1 hour
- Default size (320x180) is recommended for longer videos 