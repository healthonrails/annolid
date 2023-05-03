
def convert_time_to_frame_number(time_stamp, fps=29.97):
    """Convert string format timestamp to frame number.
    e.g. convert_time_to_frame_number('0:02:33.8',fps) -> 4585

    Args:
        time_stamp (str): string format of video timestamp
        fps (float, optional): frame rate per second. Defaults to 29.97.

    Returns:
        int: frame number 
    """
    h, m, s = time_stamp.split(':')
    seconds, milliseconds = s.split('.')
    total_seconds = int(h) * 3600 + int(m) * 60 + int(seconds)
    total_frames = int(total_seconds * fps) + int(milliseconds) * fps // 1000
    return int(total_frames)


def convert_frame_number_to_time(frame_number, fps=29.97):
    """Convert video frame_number ot timestamps. 
    e.g. convert_frame_number_to_time(555, fps) -> '00:00:18.518'

    Args:
        frame_number (int): video frame number
        fps (float, optional): Frames per second. Defaults to 29.97.

    Returns:
        str: timestamp
    """
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    time_stamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return time_stamp
