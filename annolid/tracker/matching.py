# Modified from BoT-SORT
# https://github.com/NirAharon/BoT-SORT
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from annolid.tracker import kalman_filter


def merge_matches(m1, m2, shape):
    """
    Merge two sets of matches between two sequences of items of length P.

    Parameters
    ----------
    m1 : list
        List of pairs of indices of matched items in the first sequence. Each
        pair is a tuple (i, j) indicating that item i in the first sequence
        matches item j in the second sequence.
    m2 : list
        List of pairs of indices of matched items in the second sequence. Each
        pair is a tuple (j, k) indicating that item j in the second sequence
        matches item k in the third sequence.
    shape : tuple
        A triple (O, P, Q) indicating the lengths of the three sequences.

    Returns
    -------
    tuple
        A tuple (match, unmatched_O, unmatched_Q) containing:
        - match : list of pairs of indices of matched items in the three sequences
        - unmatched_O : tuple of indices of items in the first sequence that were not matched
        - unmatched_Q : tuple of indices of items in the third sequence that were not matched
    """
    num_o, num_p, num_q = shape
    # Convert the matches to numpy arrays for efficient processing
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    # Create sparse matrices to represent the matches
    M1 = scipy.sparse.coo_matrix(
        (np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(num_o, num_p)
    )
    M2 = scipy.sparse.coo_matrix(
        (np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(num_p, num_q)
    )

    # Compute the intersection of the matches
    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))

    # Compute the unmatched items in the first and third sequences
    unmatched_O = tuple(set(range(num_o)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(num_q)) - set([j for i, j in match]))

    # Return the matches and unmatched items
    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    """
    Convert a list of indices into a list of matches based on a cost matrix.

    Parameters
    ----------
    cost_matrix : numpy.ndarray
        A 2D matrix of costs between pairs of items to be matched.
    indices : list of tuples
        A list of pairs of indices of items to be matched.
    thresh : float
        A threshold below which two items are considered a match.

    Returns
    -------
    tuple
        A tuple (matches, unmatched_a, unmatched_b) containing:
        - matches : list of pairs of indices of matched items
        - unmatched_a : tuple of indices of unmatched items from the first set
        - unmatched_b : tuple of indices of unmatched items from the second set
    """
    # Compute the cost of each matched pair
    matched_cost = cost_matrix[tuple(zip(*indices))]

    # Create a mask of matched pairs below the threshold
    matched_mask = matched_cost <= thresh

    # Get the matched pairs and the indices of unmatched items
    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    # Return the matches and unmatched items
    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> tuple:
    """
    Performs linear assignment of rows to columns in a cost matrix.

    Args:
        cost_matrix (np.ndarray): A 2D numpy array representing the cost matrix.
        thresh (float): A threshold value for the cost limit.

    Returns:
        tuple: A tuple containing three elements:
            - matches (np.ndarray): A 2D numpy array of shape (N, 2) containing the indices of matched rows and columns.
            - unmatched_a (tuple): A tuple containing the indices of unmatched rows.
            - unmatched_b (tuple): A tuple containing the indices of unmatched columns.
    """
    import lap

    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]

    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute IoU for all pairs of bounding boxes.
    :param atlbrs: list[tlbr] | np.ndarray
    :param btlbrs: list[tlbr] | np.ndarray
    :return: ious np.ndarray
    """
    from cython_bbox import bbox_overlaps as bbox_ious

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float),
    )

    return ious


def tlbr_expand(tlbr, scale=1.2):
    """
    Expand a bounding box in the form of (top, left, bottom, right).
    :param tlbr: np.ndarray
    :param scale: float
    :return: np.ndarray
    """
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU.
    :param atracks: list[STrack]
    :param btracks: list[STrack]
    :return: cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU.
    :param atracks: list[STrack]
    :param btracks: list[STrack]
    :return: cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric="cosine"):
    """
    Calculates the cost matrix for the embeddings of the tracks and detections.

    :param tracks: list[STrack]
        List of tracks to calculate the cost matrix for.
    :param detections: list[BaseTrack]
        List of detections to calculate the cost matrix for.
    :param metric: str
        Distance metric to use for calculating the cost matrix.
    :return: np.ndarray
        Cost matrix for the embeddings of the tracks and detections.
    """
    # Initialize cost matrix with zeros
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)

    # Check if cost matrix is empty
    if cost_matrix.size == 0:
        return cost_matrix

    # Convert detection and track features to numpy arrays
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)

    # Calculate the distance matrix between track and detection embeddings
    # using the specified metric
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    # cost_matrix = cost_matrix / 2.0  # Normalized features

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    Applies gating to the cost matrix to remove unlikely associations.

    :param kf: KalmanFilter
        Kalman filter used for gating.
    :param cost_matrix: np.ndarray
        Cost matrix to apply gating to.
    :param tracks: list[STrack]
        List of tracks.
    :param detections: list[BaseTrack]
        List of detections.
    :param only_position: bool
        If True, only position information is used for gating.
    :return: np.ndarray
        Cost matrix after gating.
    """
    # Check if cost matrix is empty
    if cost_matrix.size == 0:
        return cost_matrix

    # Set gating parameters based on whether only position information is used
    gating_dim = 2 if only_position else 4
    gating_threshold = kf.chi2inv95[gating_dim]

    # Convert detections to numpy array
    measurements = np.asarray([det.to_xywh() for det in detections])

    # Apply gating to cost matrix
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    Fuses motion information into a given cost matrix.

    :param kf: kalman filter object
    :param cost_matrix: np.ndarray of shape (num_tracks, num_detections)
    :param tracks: list of STrack objects
    :param detections: list of BaseTrack objects
    :param only_position: bool indicating whether to use only position or full detection information
    :param lambda_: weighting factor for motion cost vs. re-identification cost
    :return: fused cost matrix np.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix

    # Determine gating parameters
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]

    # Convert detections to measurement space
    measurements = np.asarray([det.to_xywh() for det in detections])

    # For each track, gate detections and fuse costs
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance

    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """
    Fuses intersection-over-union (IOU) information into a given cost matrix.

    :param cost_matrix: np.ndarray of shape (num_tracks, num_detections)
    :param tracks: list of STrack objects
    :param detections: list of BaseTrack objects
    :return: fused cost matrix np.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix

    # Compute re-identification similarity and IOU similarity
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist

    # Fuse similarities using a weighted average
    fuse_sim = reid_sim * (1 + iou_sim) / 2

    # Apply detection scores to the fused similarities
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_scores) / 2

    # Convert fused similarities to a cost matrix
    fuse_cost = 1 - fuse_sim

    return fuse_cost


def fuse_score(cost_matrix, detections):
    """
    Fuses detection score information into a given cost matrix.

    :param cost_matrix: np.ndarray of shape (num_tracks, num_detections)
    :param detections: list of BaseTrack objects
    :return: fused cost matrix np.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix

    # Compute IOU similarity
    iou_sim = 1 - cost_matrix

    # Apply detection scores to the IOU similarities
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores

    # Convert fused similarities to a cost matrix
    fuse_cost = 1 - fuse_sim

    return fuse_cost
