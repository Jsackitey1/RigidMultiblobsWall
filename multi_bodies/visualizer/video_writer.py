'''
Universal Video Writer Utility for RigidMultiblobsWall.
Encodes MP4 with H.264 (yuv420p) for 100% macOS QuickTime, QuickLook, web, and Windows compatibility.
Provides both a streaming context manager (VideoStreamWriter) and a batch function (save_video).
'''

import os
import cv2
import numpy as np


def _make_even(dim):
    '''Make dimension even (required for H.264 yuv420p).'''
    return dim if dim % 2 == 0 else dim - 1


def _prepare_frame(frame, even_w, even_h):
    '''Resize frame to even dimensions if needed.'''
    if frame.shape[0] != even_h or frame.shape[1] != even_w:
        frame = cv2.resize(frame, (even_w, even_h), interpolation=cv2.INTER_AREA)
    return frame


class VideoStreamWriter:
    '''
    Context manager for streaming video frames to disk one at a time.
    Avoids buffering the entire video in RAM.
    
    Usage:
        with VideoStreamWriter('output.mp4', fps=24) as writer:
            for frame in generate_frames():
                writer.write_frame(frame)
    '''

    def __init__(self, output_path, fps=24, quality=9):
        self.output_path = output_path
        self.fps = fps
        self.quality = quality
        self._writer = None
        self._backend = None  # 'imageio', 'cv2_avc1', or 'cv2_mp4v'
        self._even_w = None
        self._even_h = None
        self._frame_count = 0

        base_name, _ = os.path.splitext(output_path)
        self._mp4_path = base_name + '.mp4'
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _init_writer(self, frame):
        '''Lazily initialize the writer from the first frame's dimensions.'''
        h, w = frame.shape[:2]
        self._even_w = _make_even(w)
        self._even_h = _make_even(h)

        # Try imageio first (best compatibility)
        try:
            import imageio
            self._writer = imageio.get_writer(
                self._mp4_path,
                fps=self.fps,
                codec='libx264',
                pixelformat='yuv420p',
                quality=self.quality,
                macro_block_size=1
            )
            self._backend = 'imageio'
            return
        except Exception:
            pass

        # Fallback to OpenCV avc1
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self._writer = cv2.VideoWriter(self._mp4_path, fourcc, self.fps,
                                           (self._even_w, self._even_h))
            if self._writer.isOpened():
                self._backend = 'cv2_avc1'
                return
        except Exception:
            pass

        # Ultimate fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self._mp4_path, fourcc, self.fps,
                                       (self._even_w, self._even_h))
        self._backend = 'cv2_mp4v'

    def write_frame(self, frame):
        '''Write a single RGB frame (H, W, 3) uint8.'''
        if self._writer is None:
            self._init_writer(frame)

        frame = _prepare_frame(frame, self._even_w, self._even_h)

        if self._backend == 'imageio':
            self._writer.append_data(frame)
        else:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._writer.write(bgr)

        self._frame_count += 1

    def close(self):
        '''Finalize and close the video file.'''
        if self._writer is not None:
            if self._backend == 'imageio':
                self._writer.close()
            else:
                self._writer.release()
            self._writer = None

    @property
    def path(self):
        return self._mp4_path

    @property
    def frame_count(self):
        return self._frame_count


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

    # --- 1. Write MP4 using streaming writer ---
    if format in ['mp4', 'both']:
        with VideoStreamWriter(output_path, fps=fps, quality=quality) as writer:
            for f in frames:
                writer.write_frame(f)
        generated_files.append(writer.path)

    # --- 2. Write Animated GIF ---
    if format in ['gif', 'both']:
        gif_path = base_name + '.gif'

        # Ensure even dimensions for consistency
        first_h, first_w = frames[0].shape[:2]
        even_w = _make_even(first_w)
        even_h = _make_even(first_h)
        clean_frames = [_prepare_frame(f, even_w, even_h) for f in frames]

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
