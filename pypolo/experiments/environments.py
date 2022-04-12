from PIL import Image
import numpy as np
from pathlib import Path
from urllib import request
from skimage import transform
from matplotlib import pyplot as plt
from .visualizer import create_colorbar_ax, OOMFormatter, order_of_magnitude


class DataLoader:
    """Load data as numpy array."""

    def __init__(self, data_path: str) -> None:
        """

        Parameters
        ----------
        data_path: str
            Path to the data file.

        """
        self.data_path = data_path
        self.read_data()
        print(f"Successfully read data from {self.data_path}")
        print(f"Array \tshape:\t{self.array.shape}")
        print(f"\tmin:\t{self.array.min()}")
        print(f"\tmax:\t{self.array.max()}")
        print(f"\tdtype:\t{self.array.dtype}\n")

    def read_data(self) -> None:
        """Read data from various formats."""
        file_extension = self.data_path.split(".")[-1]
        if file_extension == "png":
            self.read_png()
        elif file_extension == "npy":
            self.read_npy()

    def read_png(self) -> None:
        """Read data from png image."""
        image = Image.open(self.data_path).convert("L")
        self.array = np.array(image).astype(np.float64)

    def read_npy(self) -> None:
        """Read data from npy file."""
        self.array = np.load(self.data_path).astype(np.float64)

    def get_data(self) -> np.ndarray:
        """Return data as numpy array."""
        return self.array


def download_environment(name):
    path = Path(f"./environments/raw/{name}.jpg")
    if not path.is_file():
        print(f"Downloading to {path}...this step might take some time.")
        request.urlretrieve(
            url="https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
            + f"{name}.SRTMGL1.2.jpg",
            filename=path,
        )
        print("Done")


def preprocess_environment(name):
    print(f"Preprocessing {name}.jpg...")
    image = Image.open(f"./environments/raw/{name}.jpg").convert("L")
    array = np.array(image).astype(np.float64)
    resized = transform.resize(array, (
        array.shape[0] // 10,
        array.shape[1] // 10,
    ))
    save_path = f"./environments/preprocessed/{name}.npy"
    np.save(save_path, resized)
    print(f"Saved to {save_path}.")


def get_environment(name):
    path = Path(f"./environments/preprocessed/{name}.npy")
    if not path.is_file():
        Path("./environments/raw").mkdir(parents=True, exist_ok=True)
        download_environment(name)
        Path("./environments/preprocessed").mkdir(parents=True, exist_ok=True)
        Path("./environments/images").mkdir(parents=True, exist_ok=True)
        preprocess_environment(name)
    data_loader = DataLoader(str(path))
    return data_loader.get_data()
