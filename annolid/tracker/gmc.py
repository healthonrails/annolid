# modified from
# https://github.com/NirAharon/BoT-SORT/blob/main/tracker/gmc.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


class GMC:
    """
    The GMC class is used for geometric model computation. 
    It includes various methods for GMC computation.

    Args:
        method (str): The method used for GMC computation.
        downscale (int): The scale of the image to be downscaled.
        verbose (str): Set to None by default.

    Attributes:
        method (str): The method used for GMC computation.
        downscale (int): The scale of the image to be downscaled.
        detector (cv2.FastFeatureDetector_create): Detector for detecting features in the image.
        extractor (cv2.ORB_create): Extractor for extracting features from the image.
        matcher (cv2.BFMatcher): Matcher for matching the features extracted from the image.
        feature_params (dict): Parameters for feature detection.
        gmcFile (open file): GMC file opened for reading GMC results.
        prevFrame (np.array): Stores the previous frame of the image.
        prevKeyPoints (np.array): Stores the previous keypoints.
        prevDescriptors (np.array): Stores the previous descriptors.
        initializedFirstFrame (bool): Set to False until the first frame is initialized.

    """

    def __init__(self, method='sparseOptFlow', downscale=2, verbose=None):
        super(GMC, self).__init__()

        self.method = method
        self.downscale = max(1, int(downscale))

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(
                nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(
                nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                             number_of_iterations, termination_eps)

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)
            # self.gmc_file = open('GMC_results.txt', 'w')

        elif self.method == 'file' or self.method == 'files':
            seqName = verbose[0]
            ablation = verbose[1]
            if ablation:
                filePath = r'tracker/GMC_files/MOT17_ablation'
            else:
                filePath = r'tracker/GMC_files/MOTChallenge'

            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]

            self.gmcFile = open(filePath + "/GMC-" + seqName + ".txt", 'r')

            if self.gmcFile is None:
                raise ValueError(
                    "Error: Unable to open GMC file in directory:" + filePath)
        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'
        else:
            raise ValueError("Error: Unknown CMC method:" + method)

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    def apply(self, raw_frame, detections=None):
        """
        Applies the selected method to the input frame and detections (if any).

        Parameters:
            raw_frame (numpy.ndarray): The input frame to apply the method to.
            detections (numpy.ndarray): The input detections to use for the method. Defaults to None.

        Returns:
            numpy.ndarray: The result of applying the selected method to the input frame and detections.

        Raises:
            ValueError: If the selected method is invalid.

        """
        # Check which method to use and apply it
        if self.method == 'orb' or self.method == 'sift':
            return self.applyFeaures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)
        elif self.method == 'none':
            return np.eye(2, 3)
        else:
            raise ValueError("Invalid method selected: {}".format(self.method))

    def applyEcc(self, raw_frame, detections=None):
        """The applyEcc method applies the Enhanced Correlation Coefficient (ECC) algorithm 
        to a frame to estimate the motion between the previous frame and the current frame.

        Parameters:

        raw_frame: a numpy array representing the current frame in BGR format.
        detections: (optional) a list of objects detected in the current frame.
        Returns:    

        H: a 2x3 numpy array representing the estimated transformation between 
        the previous and current frames. The transformation is estimated using ECC algorithm.
        Comments have been added to explain each block of code in the method.
        """
        # Get frame dimensions and convert to grayscale
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

        # Initialize transformation matrix
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            # Apply Gaussian blur and resize
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(
                frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Run the ECC algorithm. The results are stored in warp_matrix.
        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame,
                                           frame, H, self.warp_mode, self.criteria, None, 1)
        except:
            # Handle exception by setting warp as identity
            print('Warning: find transform failed. Set warp as identity')

        # Set current frame as previous frame for next iteration
        self.prevFrame = frame.copy()

        return H

    def applyFeaures(self, raw_frame, detections=None):
        """
        The method then estimates the affine transformation matrix using the inliers'
          matched keypoints using the RANSAC algorithm. If enough matching points are not found,
            a warning message is printed. Finally, the previous frame's information is updated, 
            and the transformation matrix H is returned.
        Args:
        raw_frame (numpy.ndarray): the raw frame to be used for feature detection and extraction.
        detections (list, optional): a list of bounding box coordinates (x1, y1, x2, y2) to mask the image region.

        Returns:
        H (numpy.ndarray): a 2x3 transformation matrix that aligns the previous frame and the current frame.
        """

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(
                frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # find the keypoints
        mask = np.zeros_like(frame)
        # mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
        mask[int(0.02 * height): int(0.98 * height),
             int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(
            self.prevDescriptors, descriptors, 2)

        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                   prevKeyPointLocation[1] - currKeyPointLocation[1])

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                        (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * \
            stdSpatialDistances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Draw the keypoint matches on the output image
        if 0:
            matches_img = np.hstack((self.prevFrame, frame))
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.prevFrame, 1)
            for m in goodMatches:
                prev_pt = np.array(
                    self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))

                matches_img = cv2.line(
                    matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                matches_img = cv2.circle(
                    matches_img, prev_pt, 2, tuple(color), -1)
                matches_img = cv2.circle(
                    matches_img, curr_pt, 2, tuple(color), -1)

            plt.figure()
            plt.imshow(matches_img)
            plt.show()

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(
                prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySparseOptFlow(self, raw_frame, detections=None):
        """
        Apply sparse optical flow on a single frame.

        Args:
            raw_frame (numpy array): The raw frame to apply sparse optical flow on.
            detections (list of objects, optional): A list of objects detected in the frame.

        Returns:
            numpy array: A 2x3 transformation matrix H.

        """
        t0 = time.time()

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(
                frame, (width // self.downscale, height // self.downscale))

        # Find keypoints using the Shi-Tomasi method
        keypoints = cv2.goodFeaturesToTrack(
            frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Find correspondences between the current and previous frame using Lucas-Kanade optical flow
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(
            self.prevFrame, frame, self.prevKeyPoints, None)

        # Filter out keypoints with a bad status value
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find affine transformation using RANSAC algorithm
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliers = cv2.estimateAffinePartial2D(
                prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store current frame data for use in the next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        t1 = time.time()

        # gmc_line = str(1000 * (t1 - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)

        return H

    def applyFile(self, raw_frame, detections=None):
        """
        Applies the geometric transformation described by the next line in the GMC file to the given frame.

        Args:
            raw_frame (numpy.ndarray): The input frame to transform.
            detections (None): Ignored argument. Kept for compatibility with parent class.

        Returns:
            numpy.ndarray: The affine transformation matrix that describes the geometric transformation.

        """
        # Read the next line in the GMC file
        line = self.gmcFile.readline()
        tokens = line.split("\t")

        # Parse the transformation matrix from the line
        H = np.eye(2, 3, dtype=np.float_)
        H[0, 0] = float(tokens[1])
        H[0, 1] = float(tokens[2])
        H[0, 2] = float(tokens[3])
        H[1, 0] = float(tokens[4])
        H[1, 1] = float(tokens[5])
        H[1, 2] = float(tokens[6])

        return H
