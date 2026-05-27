#!/usr/bin/env python3
import subprocess
import sys

examples = [
    "amplify",
    "amplify-onnx",
    "bevy_basic_ball_and_stick",
    "bevy_basic_putty",
    "bevy_basic_snapshot",
    "bevy_basic_spheres",
    "bevy_screenshot",
    "esm2",
    "esm2-onnx",
    "esm2-onnx-candle",
    "esmc",
    "ligandmpnn-onnx",
    "ligandmpnn-wonnx",
    "simple",
    "simple_02"
]

for example in examples:
    print(f"\n\n{'='*50}")
    print(f"Running example: {example}")
    print(f"{'='*50}\n")

    try:
        # Run the command and capture output
        process = subprocess.run(
            ["cargo", "run", "--example", example, "--features", "metal"],
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running example {example}: {e}")
        # Uncomment the next line if you want to stop on first error
        # sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(1)
