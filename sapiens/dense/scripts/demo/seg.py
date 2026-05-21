#!/usr/bin/env python3
"""
Run body-part segmentation on a directory of images.
Converted from bash script to Python with argparse.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run body-part segmentation on a directory of images'
    )

    # Input/Output
    parser.add_argument(
        '--input',
        type=str,
        default='/home/gj/Projects/2026.04.30-sapiens2/demo_img',
        help='Input directory containing images or image/ subdirectory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: {input}/sapiens_seg)'
    )

    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='sapiens2_0.4b',
        # default='sapiens2_5b',
        choices=['sapiens2_0.4b', 'sapiens2_0.8b', 'sapiens2_1b', 'sapiens2_5b'],
        help='Model size variant'
    )
    parser.add_argument(
        '--checkpoint-root',
        type=str,
        default='/home/gj/Projects/2026.04.30-sapiens2/sapiens2_host',
        help='Root directory for model checkpoints'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='shutterstock_goliath',
        help='Dataset name for model config'
    )

    # Execution
    parser.add_argument(
        '--jobs-per-gpu',
        type=int,
        default=1,
        help='Number of jobs per GPU'
    )
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=[0],
        help='GPU IDs to use (e.g., 0 1 2 3)'
    )
    parser.add_argument(
        '--class-palette',
        type=str,
        default='dome29',
        help='Class palette name'
    )

    return parser.parse_args()


def find_images_directory(input_path):
    """Find images directory: {input}/images, {input}/image, or {input} itself."""
    for subdir in ['images', 'image']:
        candidate = os.path.join(input_path, subdir)
        if os.path.isdir(candidate):
            return candidate
    return input_path


def get_image_files(directory):
    """Get all image files from directory."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    images = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(extensions):
            images.append(os.path.join(directory, f))
    return images


def distribute_workload(images, total_jobs):
    """Distribute images across jobs evenly."""
    num_images = len(images)
    images_per_file = num_images // total_jobs
    extra_images = num_images % total_jobs

    image_lists = []
    current_idx = 0
    for i in range(total_jobs):
        count = images_per_file + (1 if i < extra_images else 0)
        image_lists.append(images[current_idx:current_idx + count])
        current_idx += count

    return image_lists


def main():
    args = parse_args()

    # Resolve input, checkpoint_root, and output to absolute paths based
    # on the current working directory before we change directories.
    initial_cwd = os.getcwd()
    args.input = os.path.abspath(args.input)
    args.checkpoint_root = os.path.abspath(args.checkpoint_root) if args.checkpoint_root else args.checkpoint_root
    if args.output is not None:
        args.output = os.path.abspath(args.output)

    # Navigate to the script's parent directory (sapiens/sapiens/dense/scripts/demo)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two directories to get to sapiens root
    sapiens_root = os.path.abspath(os.path.join(script_dir, '../..'))

    # Change working directory to sapiens root
    os.chdir(sapiens_root)
    print(f"Changed working directory to: {sapiens_root}")

    # Set output directory
    if args.output is None:
        args.output = os.path.join(args.input, 'sapiens_seg')

    # Determine images directory
    images_dir = find_images_directory(args.input)
    if not os.path.isdir(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        sys.exit(1)

    # Find all images
    image_files = get_image_files(images_dir)
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in {images_dir}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Build model paths
    sapiens_checkpoint_root = args.checkpoint_root
    model_name = args.model_name
    checkpoint = os.path.join(
        sapiens_checkpoint_root,
        'seg',
        f'{model_name}_seg.safetensors'
    )

    model_str = f'{model_name}_seg_{args.dataset}-1024x768'
    config_file = f'configs/seg/{args.dataset}/{model_str}.py'
    # Use absolute path for run_file since we changed directory
    run_file = os.path.join(sapiens_root, 'tools/vis/vis_seg.py')

    # Validate paths
    if not os.path.exists(config_file):
        print(f"Warning: Config file not found at {config_file}")

    if not os.path.exists(run_file):
        print(f"Error: vis_seg.py not found at {run_file}")
        print("Make sure this script is placed in the correct location.")
        print(f"Expected location: {os.path.join(sapiens_root, 'tools/vis/vis_seg.py')}")
        sys.exit(1)

    if not os.path.exists(checkpoint):
        print(f"Warning: Checkpoint not found at {checkpoint}")
        print("Continuing anyway...")

    # Distribute workload
    total_jobs = args.jobs_per_gpu * len(args.gpu_ids)
    image_batches = distribute_workload(image_files, total_jobs)

    print(f"Distributing {len(image_files)} images into {total_jobs} jobs.")

    # Write temporary text files and run jobs
    temp_dir = os.path.join(images_dir, 'temp_paths')
    os.makedirs(temp_dir, exist_ok=True)
    temp_files = []
    processes = []

    for i, batch in enumerate(image_batches):
        # Write batch to temporary text file
        temp_file = os.path.join(temp_dir, f'image_paths_{i + 1}.txt')
        temp_files.append(temp_file)

        with open(temp_file, 'w') as f:
            for img_path in batch:
                f.write(f'{img_path}\n')

        if not batch:
            continue

        # Assign GPU
        gpu_id = args.gpu_ids[i % len(args.gpu_ids)]

        # Build command - use absolute paths
        cmd = [
            sys.executable,  # Uses the current Python interpreter
            run_file,
            config_file,
            checkpoint,
            '--save_pred',
            '--input', temp_file,
            '--output', args.output
        ]

        # Set environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'

        print(f"Job {i + 1}: GPU {gpu_id}, {len(batch)} images")
        print(f"  Command: {' '.join(cmd)}")

        if total_jobs > 1:
            # Run in background
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=sapiens_root
            )
            processes.append(process)
        else:
            # Run in foreground
            result = subprocess.run(cmd, env=env, cwd=sapiens_root)
            if result.returncode != 0:
                print(f"Error: Job {i + 1} failed with return code {result.returncode}")

    # Wait for all background processes
    for i, process in enumerate(processes):
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Job {i + 1} failed:")
            print(f"  stdout: {stdout.decode()}")
            print(f"  stderr: {stderr.decode()}")

    # Clean up temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Remove temp directory if empty
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    print(f"All jobs completed.")
    print(f"Output directory: {args.output}")


if __name__ == '__main__':
    main()