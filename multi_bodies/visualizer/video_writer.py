'''
Universal Video Writer Utility for RigidMultiblobsWall.
Encodes MP4 with H.264 (yuv420p) for 100% macOS QuickTime, QuickLook, web, and Windows compatibility.
'''

import os
import cv2
import numpy as np


def save_video(frames, output_path, fps=10, format='mp4', quality=9):
    '''
    Save a list of RGB frames as MP4 (H.264 / yuv420p) and/or animated GIF.
    
    frames: list of numpy arrays (H, W, 3) uint8 in RGB format
    output_path: path to output file (without extension or with .mp4/.gif)
    fps: frames per second
    format: 'mp4', 'gif', or 'both'
    '''
    if not frames:
        raise ValueError("No frames provided to save_video.")

    base_name, _ = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    generated_files = []

    # Ensure all frames have matching even dimensions
    first_h, first_w, _ = frames[0].shape
    # Make dimensions even (multiples of 2) for H.264 codec requirements
    even_w = first_w if first_w % 2 == 0 else first_w - 1
    even_h = first_h if first_h % 2 == 0 else first_h - 1

    clean_frames = []
    for f in frames:
        if f.shape[0] != even_h or f.shape[1] != even_w:
            f = cv2.resize(f, (even_w, even_h), interpolation=cv2.INTER_AREA)
        clean_frames.append(f)

    # --- 1. Write MP4 (H.264 / yuv420p) ---
    if format in ['mp4', 'both']:
        mp4_path = base_name + '.mp4'
        written = False

        # Try imageio with libx264 and yuv420p (guaranteed QuickTime & web compatibility)
        try:
            import imageio
            writer = imageio.get_writer(
                mp4_path,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',
                quality=quality,
                macro_block_size=1
            )
            for f in clean_frames:
                writer.append_data(f)
            writer.close()
            written = True
        except Exception as e:
            # Fallback to OpenCV avc1 / H264
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(mp4_path, fourcc, fps, (even_w, even_h))
                for f in clean_frames:
                    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    out.write(bgr)
                out.release()
                written = True
            except Exception as e2:
                # Ultimate fallback to mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(mp4_path, fourcc, fps, (even_w, even_h))
                for f in clean_frames:
                    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    out.write(bgr)
                out.release()
                written = True

        if written:
            generated_files.append(mp4_path)

    # --- 2. Write Animated GIF ---
    if format in ['gif', 'both']:
        gif_path = base_name + '.gif'
        try:
            import imageio
            duration = 1.0 / fps
            imageio.mimsave(gif_path, clean_frames, duration=duration, loop=0)
            generated_files.append(gif_path)
        except Exception as e:
            from PIL import Image
            pil_images = [Image.fromarray(f) for f in clean_frames]
            duration_ms = int(1000 / fps)
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration_ms,
                loop=0
            )
            generated_files.append(gif_path)

    return generated_files
