def convert_time_to_frame_number(time_stamp, fps=29.97):
    """Convert string format timestamp to frame number.
    e.g. convert_time_to_frame_number('0:02:33.8',fps) -> 4585

    Args:
        time_stamp (str): string format of video timestamp
        fps (float, optional): frame rate per second. Defaults to 29.97.

    Returns:
        int: frame number
    """
    h, m, s = str(time_stamp).strip().split(":")
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    total_frames = int(total_seconds * float(fps))
    return int(total_frames)


def convert_frame_number_to_time(frame_number, fps=29.97):
    """Convert video frame_number to timestamps.
    e.g. convert_frame_number_to_time(555, fps) -> '00:00:18.518'

    Args:
        frame_number (int): video frame number
        fps (float, optional): Frames per second. Defaults to 29.97.

    Returns:
        str: timestamp
    """
    fps_val = float(fps) if fps else 0.0
    if fps_val <= 0:
        return "00:00:00.000"
    total_seconds = max(0.0, float(frame_number) / fps_val)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    time_stamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return time_stamp


def convert_timestamp_to_seconds(timestamp):
    """
    Convert a timestamp in the format '00:00:01.368' to seconds.

    Args:
        timestamp (str): The timestamp in the format 'HH:MM:SS.MMM'.

    Returns:
        float: The equivalent number of seconds.

    Examples:
        >>> convert_timestamp_to_seconds('00:00:01.368')
        1.368
        >>> convert_timestamp_to_seconds('01:23:45.678')
        5025.678
    """
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    return total_seconds
