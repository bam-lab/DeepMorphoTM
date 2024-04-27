import json
import os
import torch


model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"


def load_experiment_specifications(experiment_directory, exp_No):

    filename = os.path.join(experiment_directory, 'specs_' + str(exp_No) + '.json')

    if not os.path.isfile(filename):
        raise FileNotFoundError("No experiment specifications file: '{}'".format(filename))

    return json.load(open(filename))


def get_model_params_dir(experiment_dir, exp_No, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir + '_' + str(exp_No))

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, exp_No, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir + '_' + str(exp_No))

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def save_model(experiment_directory, exp_No, filename, model, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, exp_No, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_params_dir, filename),
    )

def load_model(experiment_directory, exp_No, checkpoint, model, strict=True, map_location=torch.device('cuda')):

    filename = os.path.join(
        get_model_params_dir(experiment_directory, exp_No, True), checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename, map_location=map_location)
    model.load_state_dict(data["model_state_dict"], strict)

    return data["epoch"]

def save_optimizers(experiment_directory, exp_No, filename, optimizer, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, exp_No, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def save_latest(experiment_directory, exp_No, epoch, models, optimizers):
    save_model(experiment_directory, exp_No,  "latest.pth", models, epoch)
    save_optimizers(experiment_directory, exp_No, "latest.pth", optimizers, epoch)


def save_checkpoints(experiment_directory, exp_No, epoch, models, optimizers):
    save_model(experiment_directory, exp_No, str(epoch) + ".pth", models, epoch)
    save_optimizers(experiment_directory, exp_No, str(epoch) + ".pth", optimizers, epoch)
