# Dataset Format Reference

Common dataset loading patterns by framework. Use this to understand what format the training code expects.

## PyTorch

### ImageFolder
```python
from torchvision.datasets import ImageFolder
dataset = ImageFolder(root="./data/train", transform=transform)
```
**Expected structure:**
```
data/train/
  class_a/
    img001.jpg
    img002.png
  class_b/
    img003.jpg
```
**Validation:** Subdirectories exist, each contains image files (.jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp)

### CIFAR-10/100 (auto-download)
```python
torchvision.datasets.CIFAR10(root="./data", download=True)
```
**No preparation needed.** Data is auto-downloaded. Requires internet access on first run.

### MNIST/FashionMNIST (auto-download)
```python
torchvision.datasets.MNIST(root="./data", download=True)
```
**No preparation needed.** Data is auto-downloaded.

### Custom Dataset with CSV
```python
class MyDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
```
**Expected:** CSV file with columns matching what `__getitem__` reads. Check column names and types.

### Custom Dataset with PIL images
```python
class MyDataset(Dataset):
    def __init__(self, manifest_path):
        self.entries = json.load(open(manifest_path))
    def __getitem__(self, idx):
        img = PIL.Image.open(self.entries[idx]["path"])
```
**Expected:** JSON/CSV manifest listing image paths + labels. All referenced image files must exist.

### Pre-processed tensors
```python
data = torch.load("processed_data.pt")
```
**Expected:** `.pt` or `.pth` file containing tensor data.

## TensorFlow / Keras

### TFRecord
```python
dataset = tf.data.TFRecordDataset(["train.tfrecord"])
```
**Expected:** `.tfrecord` or `.tfrecords` files.

### image_dataset_from_directory
```python
tf.keras.preprocessing.image_dataset_from_directory("data/train")
```
**Expected:** Same as ImageFolder — subdirectories per class with images.

### CsvDataset
```python
tf.data.experimental.CsvDataset("train.csv", ...)
```
**Expected:** CSV file with proper headers/types.

## HuggingFace Datasets

### load_dataset
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
# or from local files:
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
# or from a local directory:
dataset = load_dataset("imagefolder", data_dir="./images")
```
**Expected:** Depends on the dataset type. For named datasets (e.g., "imdb"), data is auto-downloaded. For local files, the paths must exist.
**Validation:** Check if `datasets` package is installed. For local data, validate file paths. For remote datasets, ensure internet access.
**Note:** HuggingFace datasets are stored in Arrow format (`.arrow` files) in a cache directory. No special data preparation needed beyond ensuring the `datasets` package is installed.

## JAX / Flax

### TFDS (TensorFlow Datasets with JAX)
```python
import tensorflow_datasets as tfds
dataset = tfds.load('mnist', split='train', as_supervised=True)
```
**No special preparation needed.** Uses TFDS which auto-downloads. Requires `tensorflow-datasets` package.

### NumPy arrays (common in JAX)
```python
import jax.numpy as jnp
import numpy as np
data = jnp.array(np.load("data.npy"))
```
**Expected:** Same as NumPy format — `.npy` or `.npz` files.

## General Formats

### HDF5
```python
import h5py
f = h5py.File("data.h5", "r")
```
**Expected:** `.h5` or `.hdf5` file. Validate that expected dataset keys exist inside.

### NumPy arrays
```python
data = numpy.load("data.npy")  # or data.npz
```
**Expected:** `.npy` (single array) or `.npz` (multiple arrays) file.

### Parquet
```python
df = pd.read_parquet("data.parquet")
```
**Expected:** `.parquet` file.

## Common Preparation Steps

| Mismatch | Resolution |
|----------|------------|
| Flat image dir → ImageFolder | Create class subdirs, move/symlink images |
| Single dataset → train/val split | Create train/ and val/ with random split |
| CSV with wrong columns | Create new CSV with renamed/reordered columns |
| Images + labels CSV → ImageFolder | Create class subdirs, symlink images by label |
| Multiple small files → single file | Concatenate CSVs, merge HDF5 datasets |
