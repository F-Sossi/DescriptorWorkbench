#!/usr/bin/env python3
"""
KeyNet Single Image Detection Script

This script is called by the C++ KeynetDetector class to extract keypoints
from a single image using Kornia's KeyNet implementation.

The script takes an input image, processes it with KeyNet, and outputs
keypoints in CSV format for the C++ code to read back.
"""

import argparse
import sys
import numpy as np
import cv2
import torch
import kornia as K
import kornia.feature as KF
from pathlib import Path
import csv

def detect_keypoints(image_path: str, max_keypoints: int = 2000, device: str = "auto") -> list:
    """
    Detect keypoints using KeyNet

    Args:
        image_path: Path to input image
        max_keypoints: Maximum number of keypoints to detect
        device: Device to use ("auto", "cuda", "cpu")

    Returns:
        List of keypoint dictionaries with x, y, size, angle, response
    """
    # Set device
    if device == "auto":
        device = K.utils.get_cuda_device_if_available()
    else:
        device = torch.device(device)

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to PyTorch tensor [1, 1, H, W]
    image_tensor = K.image_to_tensor(image, keepdim=False).float()
    if len(image_tensor.shape) == 4:  # Remove extra dimension if present
        image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.unsqueeze(0).to(device) / 255.0

    # Initialize KeyNet detector with MultiResolutionDetector wrapper
    keynet = KF.MultiResolutionDetector(
        model=KF.KeyNet(),
        num_features=max_keypoints
    ).eval().to(device)

    # Detect keypoints
    with torch.no_grad():
        lafs, responses = keynet(image_tensor)

    # Convert LAFs to keypoint format
    keypoints = []
    lafs = lafs.cpu()[0]  # Remove batch dimension
    responses = responses.cpu()[0].squeeze(-1)  # Remove batch dimension and last dim

    for i in range(lafs.shape[0]):
        # Extract position (translation part of affine transform)
        x = float(lafs[i, 0, 2])
        y = float(lafs[i, 1, 2])

        # Extract scale (from affine transform matrix)
        scale_x = torch.norm(lafs[i, :, 0]).item()
        scale_y = torch.norm(lafs[i, :, 1]).item()
        size = (scale_x + scale_y) / 2.0 * 2.0  # Convert to OpenCV size convention

        # Extract angle (rotation)
        angle = torch.atan2(lafs[i, 1, 0], lafs[i, 0, 0]).item() * 180.0 / np.pi

        # Response value
        response = float(responses[i])

        keypoints.append({
            'x': x,
            'y': y,
            'size': size,
            'angle': angle,
            'response': response
        })

    return keypoints

def main():
    parser = argparse.ArgumentParser(description="KeyNet keypoint detection for single image")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--max_keypoints", type=int, default=2000, help="Maximum keypoints")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device")

    args = parser.parse_args()

    try:
        # Detect keypoints
        keypoints = detect_keypoints(args.input, args.max_keypoints, args.device)

        # Write to CSV
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = ['x', 'y', 'size', 'angle', 'response']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for kp in keypoints:
                writer.writerow(kp)

        # Success - C++ will check return code
        sys.exit(0)

    except Exception as e:
        print(f"KeyNet detection error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()