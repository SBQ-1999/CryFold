#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1
if [ ! -f "./CryFold/checkpoint/CryNet.pth" ] || [ ! -f "./CryFold/checkpoint/SimpleUnet.pth" ]; then
  echo "Downloading the required weight files for CryFold:"
  wget https://yanglab.qd.sdu.edu.cn/CryFold/download/checkpoints1.zip --no-check-certificate
  unzip checkpoints1.zip
  rm -f checkpoints1.zip
  if [ ! -f "./CryFold/checkpoint/CryNet.pth" ] || [ ! -f "./CryFold/checkpoint/SimpleUnet.pth" ]; then
      echo "Please manually download the weight file from https://yanglab.qd.sdu.edu.cn/CryFold/download/checkpoints1.zip, and download it to the checkpoint folder within the CryFold directory."
      exit 1
  fi
fi
echo "Detected weight files exist"
is_env_installed=$(conda info --envs | grep CryFold -c)
if [[ "${is_env_installed}" == "0" ]];then
  echo "Deploying conda environment..."
  cuda_version=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+')
  cuda_major_version=$(echo $cuda_version | cut -d. -f1)
  if [ "$cuda_major_version" -ge 11 ]; then
      echo "Detected CUDA version is above 11"
      conda env create -f linux.yml
  else
      echo "Detected CUDA version is below 11"
      conda env create -f linux-cu102.yml
  fi
else
  echo "Detected an existing CryFold environment, exiting installation";
  exit 1;
fi

if [[ `command -v activate` ]]
then
  source `which activate` CryFold
else
  conda activate CryFold
fi
  
# Check to make sure CryFold is activated
if [[ "${CONDA_DEFAULT_ENV}" != "CryFold" ]]
then
  echo "Could not run conda activate CryFold, please check the errors";
  exit 1;
fi

python_exc="${CONDA_PREFIX}/bin/python"

$python_exc setup.py install
echo "done!"
