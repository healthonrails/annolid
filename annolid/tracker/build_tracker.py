def build_tracker(tracker_type='botsort'):
    """
    Returns an instance of a tracking algorithm, given its name.

    Args:
        tracker_type (str): The name of the tracker to use.

    Returns:
        Tracker: An instance of the selected tracker algorithm.

    Raises:
        ValueError: If an invalid tracker name is provided.
    """
    if tracker_type == 'botsort':
        # Import the BoTSORT tracking algorithm
        from annolid.tracker.bot_sort import BoTSORT

        # Instantiate the tracker with the desired parameters
        botsort = BoTSORT(
            track_high_thresh=0.33824964456239337,
            track_low_thresh=0.1,
            new_track_thresh=0.21144301345190655,
            track_buffer=60,
            match_thresh=0.22734550911325851,
            proximity_thresh=0.5945380911899254,
            appearance_thresh=0.4818211117541298,
            cmc_method='sparseOptFlow',
            frame_rate=30,
            lambda_=0.9896143462366406
        )

        # Return the tracker instance
        return botsort

    # If an invalid tracker name is provided, raise an error
    raise ValueError(f"Invalid tracker type '{tracker_type}' provided.")
