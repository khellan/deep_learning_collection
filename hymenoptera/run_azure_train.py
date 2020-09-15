import argparse

from azureml.core import Dataset, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.workspace import Workspace
from azureml.train.dnn import PyTorch

CLUSTER_NAME = "gpu-cluster"
PROJECT_FOLDER = "./hymenoptera"


def get_compute_target(workspace, priority):
    try:
        compute_target = ComputeTarget(workspace=workspace, name=CLUSTER_NAME)
        print("Using existing compute target")
        created = False
    except ComputeTargetException:
        print("Creating a new compute target")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_NC6", max_nodes=1, vm_priority=priority
        )
        compute_target = ComputeTarget.create(workspace, CLUSTER_NAME, compute_config)
        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20
        )
        created = True
    return compute_target, created


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", type=str)
    parser.add_argument("--resource-group", type=str)
    parser.add_argument("--workspace-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step-size", type=int, default=7)
    args = parser.parse_args()

    workspace = Workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
    )
    compute_target, compute_target_created = get_compute_target(
        workspace, "lowpriority"
    )
    dataset = Dataset.get_by_name(workspace=workspace, name=args.dataset_name)
    data_directory = dataset.as_mount()
    experiment = Experiment(workspace, name=args.experiment_name)
    script_params = {
        "--action": "final_layer",
        "--epochs": args.epochs,
        "--learning-rate": args.learning_rate,
        "--gamma": args.gamma,
        "--momentum": args.momentum,
        "--step-size": args.step_size,
        "--environment": "azure",
        "--model-dir": "./outputs",
        "--data-dir": data_directory,
    }
    estimator = PyTorch(
        source_directory="hymenoptera",
        script_params=script_params,
        compute_target=compute_target,
        entry_script="train.py",
        use_gpu=True,
        pip_packages=["azureml-dataprep[pandas,fuse]", "azureml-mlflow"],
    )
    run = experiment.submit(estimator)
    run.wait_for_completion(show_output=True)

    if compute_target_created:
        print("Deleting compute target")
        compute_target.delete()
