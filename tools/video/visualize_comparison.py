#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import json
import logging

# Placeholder for the new function, will be defined properly below
def get_video_properties(video_path: Path) -> dict | None:
    """
    Extracts video properties (width, height, FPS) using ffprobe.
    """
    ffprobe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'json', str(video_path)
    ]
    try:
        process = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False)
        if process.returncode != 0 or not process.stdout:
            logging.error(f"ffprobe failed for {video_path}: {process.stderr.strip()}")
            return None

        data = json.loads(process.stdout)
        if not data.get('streams') or not isinstance(data['streams'], list) or len(data['streams']) == 0:
            logging.error(f"ffprobe output for {video_path} missing 'streams' list or list is empty.")
            return None
        
        stream_data = data['streams'][0]
        width = stream_data.get('width')
        height = stream_data.get('height')
        r_frame_rate_str = stream_data.get('r_frame_rate')

        if width is None or height is None or r_frame_rate_str is None:
            logging.error(f"Missing width, height, or r_frame_rate in ffprobe output for {video_path}.")
            return None

        # Convert r_frame_rate to float
        if '/' in r_frame_rate_str:
            num, den = map(float, r_frame_rate_str.split('/'))
            if den == 0: # Avoid division by zero
                logging.error(f"Invalid r_frame_rate denominator (0) for {video_path}: {r_frame_rate_str}")
                return None
            fps = num / den
        else:
            fps = float(r_frame_rate_str)
            
        return {'width': int(width), 'height': int(height), 'fps': float(fps)}

    except json.JSONDecodeError:
        logging.error(f"Failed to parse ffprobe JSON output for {video_path}.")
        return None
    except ValueError as e:
        logging.error(f"Failed to convert video properties for {video_path}: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors during ffprobe processing
        logging.error(f"An unexpected error occurred while processing {video_path} with ffprobe: {e}")
        return None

def main(exp1_dir: Path, exp2_dir: Path, output_dir: Path, exp1_name: str = "Exp 1", exp2_name: str = "Exp 2", input_name: str = "Input"):
    """
    Main function to compare two experiments and visualize the results.
    """
    logging.info(f"Starting video discovery process.")
    logging.info(f"Experiment 1 directory: {exp1_dir}")
    logging.info(f"Experiment 2 directory: {exp2_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Using labels: '{input_name}', '{exp1_name}', and '{exp2_name}'")

    for item1 in exp1_dir.iterdir():
        if not item1.is_dir():
            logging.debug(f"Skipping non-directory item in exp1_dir: {item1.name}")
            continue

        video_name = item1.name
        logging.info(f"Processing video directory: {video_name}")

        exp1_video_dir = item1 # item1 is exp1_dir / video_name
        exp2_video_dir = exp2_dir / video_name

        # Requirement 3.f: Check if the corresponding video_name subdirectory exists in exp2_dir
        if not exp2_video_dir.exists() or not exp2_video_dir.is_dir():
            logging.warning(f"Skipping {video_name}: Corresponding directory not found or is not a directory in {exp2_dir} for {video_name}")
            continue

        # Requirement 3.a: Define paths for the six required video files
        exp1_input_vid = exp1_video_dir / "0_input_video.mp4"
        exp1_incam_vid = exp1_video_dir / "1_incam.mp4"
        exp1_global_vid = exp1_video_dir / "2_global.mp4"
        exp2_input_vid = exp1_video_dir / "0_input_video.mp4" # Same as exp1_input_vid
        exp2_incam_vid = exp2_video_dir / "1_incam.mp4" # Now uses validated exp2_video_dir
        exp2_global_vid = exp2_video_dir / "2_global.mp4" # Now uses validated exp2_video_dir

        # Requirement 3.b: Create a list of these six Path objects
        video_paths_to_check = [
            exp1_input_vid,
            exp1_incam_vid,
            exp1_global_vid,
            exp2_input_vid, 
            exp2_incam_vid,
            exp2_global_vid,
        ]

        # Requirement 3.c & 3.e: Iterate and check files
        all_files_found = True
        for video_path in video_paths_to_check:
            if not video_path.exists() or not video_path.is_file():
                logging.warning(f"Required video file not found or is not a file for {video_name}: {video_path}") # Specific log
                logging.warning(f"Skipping {video_name}: Missing one or more video files.") # Generic log
                all_files_found = False
                break
        
        # Requirement 3.d: Log if all files found
        if all_files_found:
            try: # General try-except for processing a video set
                logging.info(f"Found all required videos for {video_name}.")
                
                video_props = get_video_properties(exp1_input_vid)
                if video_props is None:
                    # get_video_properties already logs specific error
                    logging.warning(f"Skipping {video_name} due to failure in extracting video properties from {exp1_input_vid}.")
                    continue # to the next video_name in the outer loop
                
                logging.info(f"Video properties for {video_name} ({exp1_input_vid.name}): {video_props['width']}x{video_props['height']} @ {video_props['fps']:.2f} FPS.")
                
                # 1. Define Output Path and Create Directory
                output_video_file = output_dir / f"{video_name}_comparison.mp4"
                
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logging.error(f"Failed to create output directory {output_dir}: {e}")
                    continue # Skip to next video if output dir can't be made

                # 2. Get Video Dimensions and FPS
                W = video_props['width']
                H = video_props['height']
                FPS = video_props['fps']

                # 3. Define Font Properties
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                font_size = max(24, int(H / 20))
                font_color = "white"

                # 4. Check for Font File Existence
                font_file_exists = Path(font_path).exists()
                if not font_file_exists:
                    logging.warning(f"Font file not found at {font_path}. Text overlays will be skipped if FFmpeg cannot find a default font.")

                # 5. Construct the FFmpeg Filter Complex String
                # Ensure even dimensions for inputs to hstack/vstack
                padding_size = max(int(H * 0.025), 8)  # 2.5% of video height with minimum of 8px
                
                pad_filter_part = (
                    "[0:v]pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2[in0];"
                    "[1:v]pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2[in1];"
                    "[2:v]pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2[in2];"
                    "[3:v]pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2[in3];"
                    "[4:v]pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2[in4];"
                    "[5:v]pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2[in5];"
                )
                
                # Add horizontal padding to each input
                pad_h_filter_part = (
                    f"[in0]pad=iw+{padding_size*2}:ih:{padding_size}:0:white[pad0];"
                    f"[in1]pad=iw+{padding_size*2}:ih:{padding_size}:0:white[pad1];"
                    f"[in2]pad=iw+{padding_size*2}:ih:{padding_size}:0:white[pad2];"
                    f"[in3]pad=iw+{padding_size*2}:ih:{padding_size}:0:white[pad3];"
                    f"[in4]pad=iw+{padding_size*2}:ih:{padding_size}:0:white[pad4];"
                    f"[in5]pad=iw+{padding_size*2}:ih:{padding_size}:0:white[pad5];"
                )
                
                hstack_filter_part = (
                    "[pad0][pad1][pad2]hstack=inputs=3[top_row];"
                    "[pad3][pad4][pad5]hstack=inputs=3[bottom_row];"
                )
                
                # Add vertical padding to rows
                pad_v_filter_part = (
                    f"[top_row]pad=iw:ih+{padding_size*2}:0:{padding_size}:white[padded_top_row];"
                    f"[bottom_row]pad=iw:ih+{padding_size*2}:0:{padding_size}:white[padded_bottom_row];"
                )
                
                vstack_filter_part = "[padded_top_row][padded_bottom_row]vstack=inputs=2[mosaic];"
                
                # Text drawing part - only add fontfile if it exists
                fontfile_option = f"fontfile='{font_path}':" if font_file_exists else ""

                # Enhanced text with box background
                drawtext_filter_part = (
                    f"[mosaic]drawtext=text='{input_name}':{fontfile_option}fontsize={font_size}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=({W}/2-text_w/2):y={padding_size+10},"
                    f"drawtext=text='{exp1_name}':{fontfile_option}fontsize={font_size}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=({W}+{W}/2-text_w/2):y={padding_size+10},"
                    f"drawtext=text='{exp2_name}':{fontfile_option}fontsize={font_size}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=(2*{W}+{W}/2-text_w/2):y={padding_size+10}[v]"
                )
                
                filter_complex_string = f"{pad_filter_part}{pad_h_filter_part}{hstack_filter_part}{pad_v_filter_part}{vstack_filter_part}{drawtext_filter_part}"

                # 6. Construct the Full FFmpeg Command List
                ffmpeg_cmd = ['ffmpeg', '-y']
                ffmpeg_cmd.extend(['-i', str(exp1_input_vid), 
                                   '-i', str(exp1_incam_vid), 
                                   '-i', str(exp2_incam_vid), 
                                   '-i', str(exp1_input_vid), # Input 3 (0-indexed) for bottom row's first video
                                   '-i', str(exp1_global_vid), # Input 4
                                   '-i', str(exp2_global_vid)]) # Input 5
                ffmpeg_cmd.extend(['-filter_complex', filter_complex_string])
                ffmpeg_cmd.extend(['-map', '[v]'])
                ffmpeg_cmd.extend(['-r', str(FPS)])
                ffmpeg_cmd.extend(['-c:v', 'libx264', '-crf', '23', '-preset', 'medium'])
                ffmpeg_cmd.append(str(output_video_file))

                # 7. Execute the Command
                logging.info(f"Executing FFmpeg command: {' '.join(ffmpeg_cmd)}")
                try:
                    process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
                    if process.returncode == 0:
                        logging.info(f"Successfully created comparison video: {output_video_file}")
                    else:
                        logging.error(f"FFmpeg failed for {video_name}: {process.stderr.strip() if process.stderr else 'N/A'}")
                except Exception as e:
                    logging.error(f"An exception occurred while running FFmpeg for {video_name}: {e}")
                
            except Exception as e: # Catch any other unexpected error during this video_name's processing
                logging.error(f"Unexpected error processing video {video_name}: {e}", exc_info=True)
                continue # to the next video_name in the outer loop

        else:
            # Requirement 3.e: ...and continue to the next subdirectory (implicitly done by loop structure)
            # This 'else' corresponds to 'if all_files_found:'
            # If not all_files_found, the outer loop continues to the next video_name.
            pass # No specific action needed here as warnings are logged inside the file check loop.

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Visualize comparison between two experiments.")
    parser.add_argument("--exp1_dir", type=Path, required=True, help="Directory of the first experiment.")
    parser.add_argument("--exp2_dir", type=Path, required=True, help="Directory of the second experiment.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the visualization output.")
    parser.add_argument("--exp1_name", type=str, default="Exp 1", help="Label for the first experiment. Default is 'Exp 1'.")
    parser.add_argument("--exp2_name", type=str, default="Exp 2", help="Label for the second experiment. Default is 'Exp 2'.")
    parser.add_argument("--input_name", type=str, default="Input", help="Label for the input video. Default is 'Input'.")

    args = parser.parse_args()

    main(args.exp1_dir, args.exp2_dir, args.output_dir, args.exp1_name, args.exp2_name, args.input_name)
