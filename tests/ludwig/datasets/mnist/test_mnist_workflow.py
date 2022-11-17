import gzip
import os
import shutil
import tempfile
from unittest import mock

from ludwig.datasets.mnist import Mnist


class FakeMnistDataset(Mnist):
    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)


def test_download_mnist_dataset():
    with tempfile.TemporaryDirectory() as source_dir:
        train_image_archive_filename = os.path.join(source_dir, "train-images-idx3-ubyte")
        with open(train_image_archive_filename, "w+b") as train_image_handle:
            train_image_handle.write(b"This binary string will be written as training mage data")
        with open(train_image_archive_filename, "rb") as f_in:
            with gzip.open(f"{train_image_archive_filename}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        train_labels_archive_filename = os.path.join(source_dir, "train-labels-idx1-ubyte")
        with open(train_labels_archive_filename, "w") as train_labels_handle:
            train_labels_handle.write("0")
        with open(train_labels_archive_filename, "rb") as f_in:
            with gzip.open(f"{train_labels_archive_filename}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        test_image_archive_filename = os.path.join(source_dir, "t10k-images-idx3-ubyte")
        with open(test_image_archive_filename, "w+b") as test_image_handle:
            test_image_handle.write(b"This binary string will be written as test mage data")
        with open(test_image_archive_filename, "rb") as f_in:
            with gzip.open(f"{test_image_archive_filename}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        test_labels_archive_filename = os.path.join(source_dir, "t10k-labels-idx1-ubyte")
        with open(test_labels_archive_filename, "w") as test_labels_handle:
            test_labels_handle.write("0")
        with open(test_labels_archive_filename, "rb") as f_in:
            with gzip.open(f"{test_labels_archive_filename}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        extracted_filenames = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ]

        download_urls = [
            f"file://{train_image_archive_filename}.gz",
            f"file://{train_labels_archive_filename}.gz",
            f"file://{test_image_archive_filename}.gz",
            f"file://{test_labels_archive_filename}.gz",
        ]


        config = dict(
            version=1.0,
            download_urls=download_urls,
            csv_filename=extracted_filenames,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("ludwig.datasets.base_dataset.read_config", return_value=config):
                dataset = FakeMnistDataset(tmpdir)
                assert not dataset.is_downloaded()
                assert not dataset.is_processed()
                dataset.download()

                assert dataset.is_downloaded()
