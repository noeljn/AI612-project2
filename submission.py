# Create submission file
import os
import shutil
import pkg_resources

def create_requirements_txt(filename="requirements.txt"):
    with open(filename, "w") as f:
        for dist in pkg_resources.working_set:
            f.write(f"{dist.project_name}=={dist.version}\n")
            #print(f"{dist.project_name}=={dist.version}")

if __name__ == "__main__":
    folder = '20227024'
    file_name = '20227024'

    # Create folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Copy model file
    shutil.copyfile('models/00000000_model.py', os.path.join(folder, f'{file_name}_model.py'))

    # Copy dataset file
    shutil.copyfile('data/00000000_dataset.py', os.path.join(folder, f'{file_name}_dataset.py'))

    # Copy preprocessing file
    shutil.copyfile('preprocess/00000000_preprocess.py', os.path.join(folder, f'{file_name}_preprocess.py'))

    # Copy newest checkpoint from outputs
    #checkpoint_folder = max([os.path.join('outputs', d) for d in os.listdir('outputs') if os.path.isdir(os.path.join('outputs', d))], key=os.path.getmtime)
    #checkpoint_folder = max([os.path.join(checkpoint_folder, d) for d in os.listdir(checkpoint_folder) if os.path.isdir(os.path.join(checkpoint_folder, d))], key=os.path.getmtime)
    #shutil.copyfile(f'{checkpoint_folder}/checkpoints/checkpoint_best.pt', os.path.join(folder, f'{file_name}_checkpoint.py'))
    
    create_requirements_txt(filename=os.path.join(folder, 'requirements.txt'))

