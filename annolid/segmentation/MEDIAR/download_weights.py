import os
import gdown


def download_weights(output_dir="./weights"):
    os.makedirs(output_dir, exist_ok=True)

    weights_urls = {
        "from_phase1": "https://drive.google.com/uc?id=168MtudjTMLoq9YGTyoD2Rjl_d3Gy6c_L",
        "from_phase2": "https://drive.google.com/uc?id=1JJ2-QKTCk-G7sp5ddkqcifMxgnyOrXjx"
    }

    for name, url in weights_urls.items():
        output_path = os.path.join(output_dir, f"{name}.pth")
        gdown.download(url, output_path, quiet=False)

    return {
        "model_path1": os.path.join(output_dir, "from_phase1.pth"),
        "model_path2": os.path.join(output_dir, "from_phase2.pth")
    }


# Example usage:
weights_paths = download_weights()
# model_path1 = weights_paths["model_path1"]
# model_path2 = weights_paths["model_path2"]
