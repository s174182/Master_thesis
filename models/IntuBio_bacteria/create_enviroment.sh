#Open interactive GPU node
#linuxsh

#Load preinstalled modules
module load python3/3.10.2

#Create a virtual environment for Python3
python3 -m venv ~/endpoint_segmentation

#Activate virtual environment
source ~/endpoint_segmentation/bin/activate

#If pip3 fails, use: which pip3, to make sure it is the one in the virutal environment.
#which pip3
python3 -m pip install hydra-core --upgrade
python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install -r requirements.txt
python3 -m pip install -U git+https://github.com/albumentations-team/albumentations
