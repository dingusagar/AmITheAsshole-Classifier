#! /usr/bin/env bash

BASEDIR=$(pwd)

check_git_lfs() {
  if command -v git-lfs > /dev/null; then
    echo "git-lfs is already installed."
    return 0
  else
    echo "git-lfs is not installed."
    return 1
  fi
}

install_git_lfs_mac() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on macOS. Installing git-lfs via Homebrew..."
    if ! command -v brew > /dev/null; then
      echo "Homebrew is not installed. Please install Homebrew first."
      exit 1
    fi
    brew install git-lfs
  else
    echo "Not on macOS, skipping git-lfs install. NOTE: git-lfs should be installed with Git for Windows"
  fi
}

initialize_git_lfs() {
  git lfs install
  if [ $? -eq 0 ]; then
    echo "Git LFS initialized successfully."
  else
    echo "Failed to initialize Git LFS."
    exit 1
  fi
}

# check if git-lfs is installed
if ! check_git_lfs; then
  # if not installed, check if we're on macOS and install git-lfs
  install_git_lfs_mac
  # initialize git-lfs
  initialize_git_lfs
fi

# proceed with cloning the repository and pulling the LFS data
REPO_URL="https://huggingface.co/datasets/MattBoraske/Reddit-AITA-2018-to-2022"
REPO_DIR="Reddit-AITA-2018-to-2022/data"


echo "Cloning the Hugging Face dataset repository..."
git clone "$REPO_URL"
if [ $? -ne 0 ]; then
  echo "Failed to clone the repository. Please check the URL and your network connection."
  exit 1
fi

cd "$REPO_DIR" || { echo "Failed to cd to $REPO_DIR"; exit 1; }
# pull the data file using git-lfs
echo "Pulling data files using git-lfs..."
git lfs pull
if [ $? -eq 0 ]; then
  echo "Data files pulled successfully."
else
  echo "Failed to pull data files using git-lfs."
  exit 1
fi

cd "$BASEDIR" || { echo "Failed to cd to ../" ; exit 1; } 

REPO_DIR_AITA="aita_dataset"

# clone github-data repo
git clone https://github.com/iterative/aita_dataset.git
cd aita_dataset || { echo "Failed to clone aita_dataset" ; exit 1 ; }

# find all .dvc files
DVC_FILES=$(ls *.dvc)

# check if there are any .dvc files
if [ -z "$DVC_FILES" ]; then
  echo "No .dvc files found in the repository."
  exit 1
fi

# loop through each .dvc file and clean it up
for dvc_file in $DVC_FILES; do
  echo "Processing $dvc_file..."
  
  # remove metric keys
  sed -i.bak '/metric:/d' "$dvc_file"

  # check if the file contains 'md5' and 'path', or else report
  if ! grep -q "md5" "$dvc_file" || ! grep -q "path" "$dvc_file"; then
    echo "Warning: $dvc_file does not contain the needed dvc fields."
  fi
done

# cull the dataset using dvc
dvc pull

# cerify if the file was pulled successfully
if [ -f "aita_clean.csv" ]; then
  echo "aita_clean.csv pulled successfully!"
else
  echo "Failed to pull aita_clean.csv. Check if the DVC remote is configured correctly."
fi

cd $BASEDIR || { echo "Could not change directories back to $BASEDIR" ; exit 1; }

# create a files directory
FILES_DIR="files"
mkdir -p "$FILES_DIR"

# create symbolic links to the necessary files
# link all .parquet files from Reddit-AITA-2018-to-2022/data/
echo "Creating symbolic links for .parquet files from Reddit-AITA-2018-to-2022..."
for parquet_file in "$BASEDIR/$REPO_DIR/"*.parquet; do
  ln -sf "$parquet_file" "$BASEDIR/$FILES_DIR/$(basename "$parquet_file")"
done

# link aita_clean.csv from aita_dataset
echo "Creating symbolic link for aita_clean.csv..."
ln -sf "$BASEDIR/$REPO_DIR_AITA/aita_clean.csv" "$BASEDIR/$FILES_DIR/aita_clean.csv"

echo "Symbolic links created successfully in the 'files' directory."
