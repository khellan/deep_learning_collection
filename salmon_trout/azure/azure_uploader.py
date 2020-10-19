import argparse
from pathlib import Path

from azureml.core.datastore import Datastore
from azureml.core.workspace import Workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", type=str)
    parser.add_argument("--resource-group", type=str)
    parser.add_argument("--workspace-name", type=str)
    parser.add_argument("--datastore-name", type=str)
    parser.add_argument("--data-directory", type=str)
    parser.add_argument("--dataset-name", type=str)
    args = parser.parse_args()

    print(args.workspace_name)
    workspace = Workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
    )
    datastore = Datastore.get(workspace, args.datastore_name)
    local_path = Path(args.data_directory)
    for phase in ["train", "val"]:
        local_directory = str(local_path / phase)
        target_path = str(Path(args.dataset_name) / phase)
        datastore.upload(
            local_directory, target_path=target_path, show_progress=True)
