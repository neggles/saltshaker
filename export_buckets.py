import argparse
from dataloaders.filedisk_loader import ImageStore, AspectBucket
import pickle


def export():
    print("creating image store...")
    image_store = ImageStore()
    print(f"{len(image_store)} image(s).")
    print("creating aspect buckets...")
    bucket = AspectBucket(image_store, args.batch_size)
    print("writing aspect buckets to file...")
    with open(args.export_path, "wb") as f:
        pickle.dump(bucket, f)

    print("done.")
