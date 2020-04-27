import pathlib
import torch
import os
from ssd.config.defaults import cfg
from train import get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    checkpoint = pathlib.Path(cfg.OUTPUT_DIR, "model_final.pth")
    assert checkpoint.is_file()
    # Create a new directory for new training run
    new_dir = checkpoint.parent.parent
    new_dir = pathlib.Path(
        checkpoint.parent.parent,
        checkpoint.parent.stem.replace("waymo", "tdt4265")
    )
    # Copy new checkpoint
    new_dir.mkdir()
    new_checkpoint_path = new_dir.joinpath("waymo_model.pth")
    # Read last checkpoint written
    with open(new_dir.joinpath("last_checkpoint.txt"), "w") as fp:
        fp.write(f"{new_checkpoint_path}")

    # Load last checkpoint and only transfer learn parameters from the model (not optimizer etc)
    new_checkpoint = {}
    new_checkpoint["model"] = torch.load(checkpoint)["model"]
    torch.save(new_checkpoint, str(new_checkpoint_path))
    del new_checkpoint

    # Transfer config file
    with open(args.config_file, "r") as fp:
        old_config_lines = fp.readlines()
    new_config_lines = []
    # Overwrite the dataset to be used and set the new output directory
    for line in old_config_lines:
        if line == '    TRAIN: ("waymo_train",)\n':
            old = line
            line = '    TRAIN: ("tdt4265_train",)\n'
            print(f"overwriting: {old} with {line}")
        if line == '    TEST: ("waymo_val", )\n':
            old = line
            line = '    TEST: ("tdt4265_val", )\n'
            print(f"overwriting: {old} with {line}")
        if 'MAX_ITER' in line:
            old = line
            line = '    MAX_ITER: 15000\n'
            print(f"overwriting: {old} with {line}")
        if 'LR_STEPS' in line:
            old = line
            line = '    LR_STEPS: [5000, 10000]\n'
        if 'LR:' in line:
            old = line
            line = '    LR: 2e-3\n'
            print(f"overwriting: {old} with {line}")
        if 'DATASET_DIR' in line:
            old = line
            line = 'DATASET_DIR: "datasets"\n'
            print(f"overwriting: {old} with {line}")
        if line.startswith('OUTPUT_DIR:'):
            line = f'OUTPUT_DIR: {new_dir}\n'
        
        # You might want to change some other hyperparameters, such as learning rate!
        new_config_lines.append(line)
    # Write new config file
    new_config_path = pathlib.Path(args.config_file).parent.joinpath(new_dir.stem + ".yml")
    with open(new_config_path, "w") as fp:
        fp.writelines(new_config_lines)

    print("Done transfer learning")
    print(f"Starting train with configs {new_config_path}")
    # Start training
    os.system(f"python train.py {new_config_path}")
    '''
    # Generate submission file to submit.
    os.system(f"python submit_results.py {new_config_path}")
    '''
