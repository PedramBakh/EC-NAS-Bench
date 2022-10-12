from google.cloud import storage
import os

TABULAR_BENCHMARKS = os.path.join(os.getcwd(), "ecnas", "utils", "data", "tabular_benchmarks")


def get_nasbench101_full(dest_file_path=None):
    if dest_file_path:
        dest_file = dest_file_path
    else:
        dest_file = os.path.join(TABULAR_BENCHMARKS, "nasbench_full.tfrecord")

    if os.path.exists(dest_file):
        print(f"File {dest_file} already exists.")
        return

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket("nasbench")
    blob = bucket.get_blob("nasbench_full.tfrecord")
    blob.download_to_filename(dest_file)

    print(f"Downloaded public blob nasbench_full.tfrecord from bucket nasbench to {dest_file}.")

    return


def get_nasbench101_only108(dest_file_path=None):
    if dest_file_path:
        dest_file = dest_file_path
    else:
        dest_file = os.path.join(TABULAR_BENCHMARKS, "nasbench_only108.tfrecord")

    if os.path.exists(dest_file):
        print(f"File {dest_file} already exists.")
        return

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket("nasbench")
    blob = bucket.get_blob("nasbench_only108.tfrecord")
    blob.download_to_filename(dest_file)

    print(f"Downloaded public blob nasbench_only108.tfrecord from bucket nasbench to {dest_file}.")

    return
