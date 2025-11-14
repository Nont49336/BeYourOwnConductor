# BeYourOwnConductor
This is the project for the final project of Sound and Music Computing.

## Setup Instructions
1. create a conda environment with python 3.11
    ```bash
    conda create -n byoc python=3.11
    conda activate byoc
    ```
2. Install pytorch following the instructions at https://pytorch.org/get-started/locally/ (make sure to select the right CUDA version if you have a GPU)
3. install the required packages
   ```bash
   pip install -r requirements.txt
   ```
4. install fluidsynth via conda
   ```bash
   conda install -c conda-forge fluidsynth
   ```

## Run the project
To run the project, execute the following command:
```bash
python demo_finger_conducting.py
```