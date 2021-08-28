# deepDiffusion
A Deep Neural Network-based diffusion equation solver using TensorFlow.

## Contributors
- [Sayan Adhikari](https://github.com/sayanadhikari), UiO, Norway. [@sayanadhikari](https://twitter.com/sayanadhikari)
- [Rupak Mukherjee](https://github.com/RupakMukherjee), PPPL, USA.

## Installation
### Prerequisites
1. [python3 or higher](https://www.python.org/download/releases/3.0/)
2. [git](https://git-scm.com/)
3. [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Procedure
#### Using Anaconda/Miniconda
First make a clone of the master branch using the following command
```shell
git clone https://github.com/Neural-Plasma/deepDiffusion.git
```
Then enter inside the *deepDiffusion* directory
```shell
cd deepDiffusion
```
Now create a conda environment using the given *environment.yml* file
```shell
conda env create -f environment.yml
```
Activate the conda environment
```shell
conda activate deepDiffusion
```
## Usage

Run the code using following command

#### To train the model
```
python deepDiff --train
```
#### To test the model
```
python deepDiff --test
```
