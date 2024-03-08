import threading
from collections import OrderedDict
from collections import deque


class ThreadSafeLRUCache:
    """A thread-safe and efficient Least Recently Used (LRU) cache implementation.

    This class provides a thread-safe mechanism for caching items with a fixed maximum size.
    When the cache reaches its maximum size, the least recently used item is automatically evicted
    to make room for new items.

    Usage:
    - Create an instance of ThreadSafeLRUCache with a specified maximum size.
    - Use the 'put' method to add key-value pairs to the cache.
    - Use the 'get' method to retrieve values associated with keys.
    - Use the 'contains' method to check if a key is present in the cache.

    Example:
    >>> cache = ThreadSafeLRUCache(max_size=100)
    >>> cache.put("key1", "value1")
    >>> value = cache.get("key1")
    >>> if cache.contains("key2"):
    ...     value2 = cache.get("key2")

    Attributes:
    - max_size (int): The maximum number of items that the cache can hold."""

    def __init__(self, max_size=30):
        """
        Initialize the ThreadSafeLRUCache with a maximum size.

        Args:
            max_size (int): The maximum number of items to store in the cache.
        """
        self.max_size = max_size
        self.cache_dict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        """
        Retrieve a value from the cache. Returns None if the key is not present.

        Args:
            key: The key to retrieve the value for.

        Returns:
            The value associated with the key, or None if the key is not present.
        """
        with self.lock:
            if key not in self.cache_dict:
                return None
            self.cache_dict.move_to_end(key)
            return self.cache_dict[key]

    def put(self, key, value):
        """
        Add a value to the cache. If the cache is full, the oldest item is evicted.

        Args:
            key: The key to associate with the value.
            value: The value to store in the cache.
        """
        with self.lock:
            self.cache_dict[key] = value
            self.cache_dict.move_to_end(key)
            if len(self.cache_dict) > self.max_size:
                self.cache_dict.popitem(last=False)

    def contains(self, key):
        """
        Check if the key is present in the cache.

        Args:
            key: The key to check for in the cache.

        Returns:
            True if the key is in the cache, False otherwise.
        """
        with self.lock:
            return key in self.cache_dict


class BboxCache:
    def __init__(self, max_size=100):
        self.cache = ThreadSafeLRUCache(max_size=max_size)

    def add_bbox(self, key, bbox):
        """
        Add a bounding box to the cache.

        Args:
            key: A unique identifier for the bounding box.
            bbox: The bounding box coordinates in the format (x1, y1, x2, y2).
        """
        with self.cache.lock:
            if key not in self.cache.cache_dict:
                # Create a new queue for the key if it doesn't exist
                self.cache.cache_dict[key] = deque(maxlen=self.cache.max_size)
            # Add the new bounding box to the queue associated with the key
            self.cache.cache_dict[key].append(bbox)

    def get_bboxes(self, key):
        """
        Retrieve bounding boxes associated with a key from the cache.

        Args:
            key: The unique identifier of the bounding boxes.

        Returns:
            A list of bounding box coordinates (x1, y1, x2, y2) associated with the key,
            or an empty list if the key is not present in the cache.
        """
        with self.cache.lock:
            bbox_queue = self.cache.cache_dict.get(key, [])
            # Convert the deque to a list before returning
            return list(bbox_queue)

    def contains_key(self, key):
        """
        Check if a key exists in the cache.

        Args:
            key: The unique identifier of the key.

        Returns:
            True if the key exists in the cache, False otherwise.
        """
        return self.cache.contains(key)

    def get_most_recent_bbox(self, key):
        """
        Retrieve the most recent bounding box associated with a key from the cache.

        Args:
            key: The unique identifier of the bounding boxes.

        Returns:
            The most recent bounding box coordinates (x1, y1, x2, y2) associated with the key,
            or None if the key is not present in the cache.
        """
        with self.cache.lock:
            bbox_queue = self.cache.cache_dict.get(key)
            if bbox_queue:
                # Return the most recent bounding box from the queue
                return bbox_queue[-1]
            else:
                return None
