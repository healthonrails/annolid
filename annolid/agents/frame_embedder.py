import asyncio
import logging
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Optional

import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import Vector
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from annolid.utils.annotation_store import AnnotationStoreError, load_labelme_json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
ANNOTATION_EXTENSIONS = {".json"}


class Config:
    DB_PATH = "lancedb"
    TABLE_NAME = "video_frames"
    MAX_INDEX_IMAGES = 1000


# Setup Embedding Function
registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create()


class LanceDBFrameIndexer:
    def __init__(
        self, db_path: str = Config.DB_PATH, table_name: str = Config.TABLE_NAME
    ):
        self.db_uri = db_path
        self.table_name = table_name
        self.db = None  # Lazy load the client
        self.table = None  # Lazy load the table

    async def get_async_client(self):
        if self.db is None:
            self.db = await lancedb.connect_async(self.db_uri)
        return self.db

    async def get_async_table(self):
        if self.table is None:
            db = await self.get_async_client()
            if self.table_name in await db.table_names():
                self.table = await db.open_table(self.table_name)
            else:
                self.table = await db.create_table(self.table_name, schema=LanceDBFrame)
        return self.table

    async def index_frames(self, frame_data: List[Dict[str, Any]]):
        """Indexes a list of frames with their annotation data."""
        try:
            table = await self.get_async_table()
            await table.add(frame_data)
            logger.info(f"Indexed {len(frame_data)} frames.")
        except Exception as e:
            logger.error(f"Error indexing frames: {e}")

    async def search_similar_images(self, query_image, limit=3) -> List["LanceDBFrame"]:
        """Searches the table for similar images.
        Args:
            query_image (PIL.Image.Image or np.ndarray): The query image.
            limit (int): The number of results to return.

        Returns:
            List[LanceDBFrame]: A list of pydantic objects that match the query.
        """
        try:
            table = await self.get_async_table()
            results = (
                await table.search(query_image).limit(limit).to_pydantic(LanceDBFrame)
            )
            return results
        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []

    async def optimize_table(self):
        """Optimizes the table for better performance"""
        try:
            table = await self.get_async_table()
            await table.optimize()
            logger.info("Optimization of table is done.")
        except Exception as e:
            logger.error(f"Error Optimizing table: {e}")


class LanceDBFrame(lancedb.pydantic.LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()
    flags: Optional[List[str]] = None
    caption: Optional[str] = None


class FolderMonitor(FileSystemEventHandler):
    def __init__(self, folder_path: Path, indexer: LanceDBFrameIndexer):
        self.folder_path = folder_path
        self.indexer = indexer

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            asyncio.run(self.process_new_image(file_path))

    async def process_new_image(self, file_path: Path):
        """Processes a new image and its associated annotation."""
        logger.info(f"New image detected: {file_path}")
        annotation_file = file_path.with_suffix(".json")
        if annotation_file.exists():
            try:
                annotation_data = load_labelme_json(annotation_file)
                frame_labels = list(annotation_data.get("flags", {}).keys())
                caption = annotation_data.get("caption", "")

                frame_data = [
                    {
                        "image_uri": str(file_path),
                        "flags": frame_labels,
                        "caption": caption,
                    }
                ]
            except (AnnotationStoreError, Exception) as e:
                logger.error(f"Error indexing new image:{file_path} - {e}")
        else:
            frame_data = [{"image_uri": str(file_path), "flags": [], "caption": ""}]
        await self.indexer.index_frames(frame_data)


async def index_existing_images(folder_path: Path, indexer: LanceDBFrameIndexer):
    """Indexes existing images and their associated annotations in a folder."""
    image_uris = list(folder_path.glob("*"))
    if not image_uris:
        logger.warning("No images or annotation files found in the folder.")
        return

    image_uris = [str(f) for f in image_uris if f.suffix.lower() in IMAGE_EXTENSIONS]

    sampled_uris = sample(image_uris, min(Config.MAX_INDEX_IMAGES, len(image_uris)))

    logger.info(f"Indexing {len(sampled_uris)} images from folder.")

    frame_data_list = []
    for image_file in sampled_uris:
        annotation_file = Path(image_file).with_suffix(".json")
        if annotation_file.exists():
            try:
                annotation_data = load_labelme_json(annotation_file)
                frame_data_list.append(
                    {
                        "image_uri": image_file,
                        "flags": list(annotation_data.get("flags", {}).keys()),
                        "caption": annotation_data.get("caption", ""),
                    }
                )
            except (AnnotationStoreError, Exception) as e:
                logger.error(f"Error indexing existing image: {image_file} - {e}")
        else:
            frame_data_list.append(
                {
                    "image_uri": image_file,
                    "flags": [],
                    "caption": "",
                }
            )

    if frame_data_list:
        await indexer.index_frames(frame_data_list)


async def monitor_folder(folder_path: Path, indexer: LanceDBFrameIndexer):
    """Monitors a folder for new images and indexes them."""
    event_handler = FolderMonitor(folder_path, indexer)
    observer = Observer()
    observer.schedule(event_handler, str(folder_path), recursive=False)
    observer.start()

    try:
        while True:
            # Keep the program running to monitor changes
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


async def embed_frames(folder_path: str):
    """Main function to index and monitor a folder of images."""
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        logger.error(f"Invalid folder path: {folder_path}")
        return

    indexer = LanceDBFrameIndexer()

    # Index existing images in the folder
    await index_existing_images(folder_path, indexer)

    # Start monitoring the folder for new images
    monitor_task = asyncio.create_task(monitor_folder(folder_path, indexer))
    logger.info(f"Monitoring folder: {folder_path} for new images...")

    try:
        await monitor_task
    except KeyboardInterrupt:
        logger.info("Shutting down folder monitoring.")
    finally:
        await indexer.optimize_table()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Index and monitor a folder of images and their annotations."
    )
    parser.add_argument("folder_path", help="Path to the folder containing images.")
    args = parser.parse_args()

    asyncio.run(embed_frames(args.folder_path))
