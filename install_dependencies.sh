#!/bin/bash

# Function to install dependencies in the virtual environment
install_dependencies() {

    # Step 3: Install PyTorch based on CUDA argument
    echo "Installing PyTorch with $cuda_option..."

    # Logic to handle different OS and CUDA options
    if [[ "$os" == "Mac" ]]; then
        # macOS doesn't support CUDA, install CPU version by default
        pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
    else
        if [[ "$cuda_option" == "cpu" ]]; then
            pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
        elif [[ "$cuda_option" == "cu118" ]]; then
            pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        elif [[ "$cuda_option" == "cu121" ]]; then
            pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        else
            echo "Invalid CUDA option: $cuda_option. Please use 'cpu', 'cu118', or 'cu121'."
            exit 1
        fi
    fi

    echo "Install swig"
    pip install swig

    echo "Installing gymnasium..."
    pip install gymnasium
    pip install gymnasium[box2d]

    echo "Installing hydra..."
    pip install hydra-core --upgrade
    pip install hydra_colorlog --upgrade
    pip install hydra-optuna-sweeper --upgrade
    pip install hydra-ray-launcher --upgrade

    echo "Installing tensorboard..."
    pip install tensorboard 

    echo "Installing jupyter..."
    pip install notebook
    
    echo "Installing moviepy..."
    pip install moviepy

    echo "Installing matplotlib..."
    pip install matplotlib

    

    echo "All dependencies are installed."
}

# Detect the operating system
os="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    os="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    os="Mac"
elif [[ "$OSTYPE" == "cygwin" ]]; then
    os="Windows"
elif [[ "$OSTYPE" == "msys" ]]; then
    os="Windows"
elif [[ "$OSTYPE" == "win32" ]]; then
    os="Windows"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $os"

# Check for the CUDA option argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <cpu|cu118|cu121>"
    exit 1
fi

cuda_option=$1
echo "Selected CUDA option: $cuda_option"

#Execute the function to create environment and install dependencies
install_dependencies
