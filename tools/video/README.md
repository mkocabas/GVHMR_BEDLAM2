## Video Comparison Mosaic Generator (`visualize_comparison.py`)

This script generates a video mosaic comparing results from two different experiments. For each corresponding video set found in the experiment directories, it creates a 2x3 composite video.

**Output Layout:**

The mosaic is arranged as follows:

| Column Text | Input         | Exp 1 Results | Exp 2 Results |
|-------------|---------------|---------------|---------------|
| **Row 1**   | Input Video   | In-camera     | In-camera     |
| **Row 2**   | Input Video   | Global        | Global        |

Column titles ("Input", "Exp 1", "Exp 2") are drawn at the top-center of each respective column.

### Dependencies

*   **Python 3**: The script is written in Python.
*   **FFmpeg**: Must be installed and accessible in your system's PATH. FFmpeg is used for video processing, and `ffprobe` (which is part of FFmpeg) is used to get video properties.
    *   On Ubuntu/Debian, you can install it via: `sudo apt update && sudo apt install ffmpeg`

### Input Directory Structure

The script expects a specific directory structure for the input experiment data:

*   **Experiment Directory 1 (`--exp1_dir`)**:
    ```
    <exp1_dir>/
    ├── <video_name_1>/
    │   ├── 0_input_video.mp4  (Original input video)
    │   ├── 1_incam.mp4        (In-camera results for Exp 1)
    │   └── 2_global.mp4       (Global results for Exp 1)
    ├── <video_name_2>/
    │   ├── ...
    └── ...
    ```
*   **Experiment Directory 2 (`--exp2_dir`)**:
    ```
    <exp2_dir>/
    ├── <video_name_1>/  (Must match a video_name in exp1_dir)
    │   ├── 1_incam.mp4        (In-camera results for Exp 2)
    │   └── 2_global.mp4       (Global results for Exp 2)
    ├── <video_name_2>/
    │   ├── ...
    └── ...
    ```

The script iterates through each `<video_name>` subfolder in `<exp1_dir>`. For each one, it looks for a corresponding `<video_name>` subfolder in `<exp2_dir>` and the specified video files within them.

### Usage

```bash
python tools/video/visualize_comparison.py --exp1_dir path/to/experiment1_outputs --exp2_dir path/to/experiment2_outputs --output_dir path/to/comparison_videos
```

**Arguments:**

*   `--exp1_dir`: Path to the root directory of the first experiment's video outputs.
*   `--exp2_dir`: Path to the root directory of the second experiment's video outputs.
*   `--output_dir`: Path to the directory where the generated comparison videos will be saved. Each comparison video will be named `<video_name>_comparison.mp4`.

### Notes

*   The script uses `ffprobe` to determine the resolution and FPS of the `0_input_video.mp4` from experiment 1, and uses these properties for the output mosaic.
*   A default font (`DejaVuSans-Bold`) is used for the column titles. Ensure this font is available on your system (common on Ubuntu) or modify the `font_path` variable in the script if needed. If the specified font is not found, text drawing might be skipped or use an FFmpeg default font.
*   If any of the required video files are missing for a particular `video_name`, that video set will be skipped, and a warning will be logged.
