from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path

PHASES = ["train", "val"]


def move_images(data_directory, class_name, phase, samples):
    (Path(data_directory) / phase / class_name).mkdir(parents=True)
    for file_id in samples:
        filename = f"aug_{file_id}.jpg"
        Path(f"{data_directory}/{class_name}/{filename}").rename(
            f"{data_directory}/{phase}/{class_name}/{filename}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=str)
    parser.add_argument("--classes", type=str)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()

    class_names = [class_name.strip() for class_name in args.classes.split(",")]
    for class_name in class_names:
        samples = list(range(1, args.num_samples + 1))
        train, validate = train_test_split(samples, test_size=args.test_size)
        move_images(args.data_directory, class_name, "train", train)
        move_images(args.data_directory, class_name, "val", validate)
