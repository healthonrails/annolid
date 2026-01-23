import lancedb
from annolid.agents.frame_embedder import LanceDBFrame, Config
import logging
from typing import List, Union
from PIL import Image
from annolid.core.media.video import CV2Video
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global indexer instance
indexer = None


def cv2_to_pil(frame: np.ndarray, *, input_color: str = "rgb") -> Image.Image:
    """
    Converts an OpenCV frame (NumPy array) to a PIL Image.

    Args:
        frame (np.ndarray): The image array.
        input_color (str): "rgb" (default) or "bgr".

    Returns:
        Image.Image: The converted PIL Image in RGB format.
    """
    if frame is None:
        raise ValueError("The input frame is None.")

    input_color = str(input_color or "").strip().lower()
    rgb_frame = frame
    if input_color == "bgr":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create and return a PIL Image
    return Image.fromarray(rgb_frame)


class LanceDBFrameIndexer:
    def __init__(
        self, db_path: str = Config.DB_PATH, table_name: str = Config.TABLE_NAME
    ):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = None
        self.setup_table()

    def setup_table(self):
        """Sets up the table by connecting to it if it exists in the database."""
        if self.table_name in self.db:
            self.table = self.db[self.table_name]
            logger.info(f"Connected to table '{self.table_name}' in LanceDB.")
        else:
            logger.error(f"Table '{self.table_name}' does not exist in LanceDB.")

    def search_similar_images(self, query_image, limit: int = 3) -> List[LanceDBFrame]:
        """
        Searches for similar images in the database.

        Args:
            query_image: The query image as a string or PIL Image.
            limit (int): The number of top results to retrieve.

        Returns:
            List[LanceDBFrame]: List of matched results.
        """
        if not self.table:
            logger.error("No table connected. Cannot perform search.")
            return []

        try:
            results = (
                self.table.search(query_image).limit(limit).to_pydantic(LanceDBFrame)
            )
            return results
        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []


def get_embedder() -> LanceDBFrameIndexer:
    """Ensures a single instance of the LanceDBFrameIndexer is initialized."""
    global indexer
    if indexer is None:
        indexer = LanceDBFrameIndexer()
    return indexer


def search_frames(
    query_image: Union[str, Image.Image],
    limit: int = 3,
) -> List[dict]:
    """
    Searches for frames similar to the query image.

    Args:
        query_image (Union[str, Image.Image]): File path to an image or a PIL Image object.
        limit (int): Number of results to return.

    Returns:
        List[dict]: List of dictionaries containing `image_uri`, `flags`, and `caption` for each matched result.
    """
    # Perform the search
    embedder = get_embedder()
    results = embedder.search_similar_images(query_image, limit=limit)

    # Format the results
    formatted_results = [
        {
            "image_uri": result.image_uri,
            "flags": result.flags,
            "caption": result.caption,
        }
        for result in results
    ]

    return formatted_results


def search_video(video_path, frame_skip=1, search_limit=1):
    """
    Searches for objects in video frames and prints the results.

    Args:
        video_path (str): Path to the video file.
        frame_skip (int): Number of frames to skip for faster processing (default is 1, meaning no skipping).
        search_limit (int): Number of results to return from `search_frames` (default is 1).
    """
    try:
        video_loader = CV2Video(video_path)
        num_frames = video_loader.total_frames()

        print(f"Processing video: {video_path}")
        print(f"Total frames: {num_frames}, Skipping every {frame_skip} frames.")

        for i in range(0, num_frames, frame_skip):  # Skip frames if needed
            frame = video_loader.load_frame(i)
            if frame is None:  # Handle missing frames
                print(f"Warning: Failed to load frame {i}")
                continue

            pil_img = cv2_to_pil(frame)
            results = search_frames(pil_img, limit=search_limit)

            if results:
                for res in results:
                    print(f"Frame {i}: {res['image_uri']}")
            else:
                print(f"Frame {i}: No results found.")

    except FileNotFoundError:
        print(f"Error: Video file '{video_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search for similar images in a LanceDB database."
    )
    # parser.add_argument("query_frame", help="Path to the query image.")
    parser.add_argument("video_path", help="Video path")
    args = parser.parse_args()
    search_video(args.video_path)

    # results = search_frames(args.query_frame, limit=5)
    # if results:
    #     for i, res in enumerate(results, start=1):
    #         print(f"Result {i}:")
    #         print(f"  Image URI: {res['image_uri']}")
    #         print(f"  Flags: {res['flags']}")
    #         print(f"  Caption: {res['caption']}")
    # else:
    #     logger.info("No results found.")
