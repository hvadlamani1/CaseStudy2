#!/bin/bash

#Load ENV, #export env vars
if [ -f .env ]; then
    set -a          
    source .env
    set +a

else
    echo ".env file not found!"
    exit 1
fi

# Paths and variables for public and private key
#Path is changed when running the cron job
LOCAL_DEFAULT_KEY="group_key"
LOCAL_NEW_KEY="new_key"

VM_USER="group01"
VM_HOST="paffenroth-23.dyn.wpi.edu"
VM_PORT="22001" 
VM_FRONTEND="22000" 

# The public key string you want to inject into the VM
MY_PUBLIC_KEY=$MY_PUBLIC_KEY

echo "Starting Automatic Server Recovery Check..."

#check if server is on

if ssh -i "$LOCAL_NEW_KEY" -p $VM_PORT -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "pgrep uvicorn" > /dev/null 2>&1; then
    echo "$(date): API is running normally. No recovery needed."
    exit 0
fi

echo "$(date): VM appears down or wiped. Initiating recovery protocol !"


# SSH KEY MANAGEMENT
ssh-keygen -R "[$VM_HOST]:$VM_PORT" > /dev/null 2>&1

# Push personal key
echo "Pushing personal public key to secure the VM..."
ssh -i "$LOCAL_DEFAULT_KEY" -p $VM_PORT -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "
    mkdir -p ~/.ssh && 
    chmod 700 ~/.ssh && 
    echo '$MY_PUBLIC_KEY' > ~/.ssh/authorized_keys && 
    chmod 600 ~/.ssh/authorized_keys
"

#deploy

echo "Connecting with personal key to rebuild infrastructure..."

ssh-keygen -R "[paffenroth-23.dyn.wpi.edu]:22001"  


#BACKEND DEPLOYMENT
ssh -i "$LOCAL_NEW_KEY" -p $VM_PORT -o StrictHostKeyChecking=no $VM_USER@$VM_HOST << 'EOF'
    set -e 

    # 1. Clone the GitHub Repository
    echo "Cloning the repository..."
    REPO_URL="https://github.com/hvadlamani1/CaseStudy2.git"
    REPO_DIR="CaseStudy2"
    rm -rf $REPO_DIR
    git clone $REPO_URL
    
    # 2. Install Miniconda Silently (-b means batch mode, no prompts)
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Initialize conda for this session
    source $HOME/miniconda3/bin/activate
    
    # 3. Setup Virtual Environment
    echo "Setting up the virtual environment..."
    cd $HOME/CaseStudy2/src
    
    # Use standard python venv inside the base conda environment
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    
    pip install -r requirements.txt

    # 4. Deploy the Backend
    echo "Starting Backend API..."
    # Note: Fixed the directory path from CaseStudy to CaseStudy2
    cd $HOME/CaseStudy2/src/backend
    
    # Kill existing processes just in case
    pkill -f uvicorn || true
    
    nohup uvicorn backend_api:app --host 0.0.0.0 --port 9001 > backend.log 2>&1 &
    echo "Backend deployed successfully on port 9001."

EOF

ssh-keygen -R "[paffenroth-23.dyn.wpi.edu]:22000"  


#FRONTEND DEPLOYMENT
ssh -i "$LOCAL_DEFAULT_KEY" -p $VM_FRONTEND -o StrictHostKeyChecking=no group01@$VM_HOST << 'EOF'
    set -e 

    echo "Setting up Frontend" 

    # 1. Clone the GitHub Repository
    echo "Cloning the repository..."
    REPO_URL="https://github.com/hvadlamani1/CaseStudy2.git"
    REPO_DIR="CaseStudy2"
    rm -rf $REPO_DIR
    git clone $REPO_URL
    
    # 2. Install Miniconda Silently (-b means batch mode, no prompts)
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Initialize conda for this session
    source $HOME/miniconda3/bin/activate
    
    # 3. Setup Virtual Environment
    echo "Setting up the virtual environment..."
    cd $HOME/CaseStudy2/src
    
    # Use standard python venv inside the base conda environment
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    
    pip install -r requirements.txt

    # 4. Deploy the Backend
    echo "Starting Backend API..."
    # Note: Fixed the directory path from CaseStudy to CaseStudy2
    cd $HOME/CaseStudy2/src/backend
    

    #creating gradio temp files
    mkdir -p $HOME/CaseStudy2/src/frontend/tmp
    export GRADIO_TEMP_DIR="$HOME/CaseStudy2/src/frontend/tmp"

    # 5. Deploy the Frontend
    echo "Starting Frontend UI..."
    cd $HOME/CaseStudy2/src/frontend
    pkill -f frontend_ui.py || true
    
    export HF_TOKEN=$HF_TOKEN
    nohup python frontend_ui.py > frontend.log 2>&1 &
    echo "Frontend deployed successfully on port 7001."

EOF

echo "$(date): Automated Recovery Script Finished!"
