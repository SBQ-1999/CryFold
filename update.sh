#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

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
pip uninstall CryFold
python_exc="${CONDA_PREFIX}/bin/python"

$python_exc setup.py install
echo "done!"
