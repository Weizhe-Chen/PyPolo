from pathlib import Path
from urllib import request

from PIL import Image
import numpy as np
from pathlib import Path
from urllib import request
from skimage import transform

name = "N17E073"
path = Path(f"{name}.jpg")
if not path.is_file():
    print(f"Downloading to {path}...this step might take some time.")
    request.urlretrieve(
        url="https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/" +
        f"{name}.SRTMGL1.2.jpg",
        filename=path,
    )
    print("Done")

print(f"Preprocessing {name}.jpg...")
image = Image.open(f"{name}.jpg").convert("L")
array = np.array(image).astype(np.float64)
resized = transform.resize(array, (
    array.shape[0] // 10,
    array.shape[1] // 10,
))
save_path = f"{name}.npy"
np.save(save_path, resized)
print(f"Saved to {save_path}.")
