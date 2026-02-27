import os

from annolid.utils.model_assets import ensure_cached_model_asset


class WeightDownloader:
    def __init__(self, weights_dir):
        self.weights_dir = weights_dir

    def download_weights(self, weight_url, expected_checksum, weight_file_name):
        """
        Download pretrained weights for MEDIAR models.
        """
        # Create weights directory if it doesn't exist
        os.makedirs(self.weights_dir, exist_ok=True)

        # Define the weight file path
        weight_file_path = os.path.join(self.weights_dir, weight_file_name)

        ensure_cached_model_asset(
            file_name=weight_file_name,
            url=weight_url,
            expected_md5=expected_checksum,
            cache_dir=self.weights_dir,
            quiet=False,
            fuzzy=True,
        )

        # Check if the file exists and has been downloaded successfully
        if os.path.exists(weight_file_path):
            print(f"Weight file downloaded successfully: {weight_file_name}")
        else:
            print(f"Failed to download weight file: {weight_file_name}")


# Usage
if __name__ == "__main__":
    weights_dir = "./weights"
    downloader = WeightDownloader(weights_dir)

    # Define weight URLs, expected checksums, and file names
    weight_urls = [
        "https://drive.google.com/uc?id=168MtudjTMLoq9YGTyoD2Rjl_d3Gy6c_L",
        "https://drive.google.com/uc?id=1JJ2-QKTCk-G7sp5ddkqcifMxgnyOrXjx",
    ]
    expected_checksums = [
        "e0ccb052828a9f05e21b2143939583c5",
        "a595336926767afdf1ffb1baffd5ab7f",
    ]
    weight_file_names = ["from_phase1.pth", "from_phase2.pth"]

    # Download weights for each URL
    for url, checksum, file_name in zip(
        weight_urls, expected_checksums, weight_file_names
    ):
        downloader.download_weights(url, checksum, file_name)
