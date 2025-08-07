# Instructions to Connect EC2 & Access Scripts

## EC2 Setup and Connection

### 1. Launch EC2 Instance
```bash
# Create key pair
aws ec2 create-key-pair --key-name exl-churn-key --query 'KeyMaterial' --output text > exl-churn-key.pem

# Launch instance
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --count 1 \
    --instance-type t3.medium \
    --key-name exl-churn-key \
    --security-groups default
```

### 2. Connect to EC2
```bash
# Make key file secure
chmod 400 exl-churn-key.pem

# Connect via SSH
ssh -i "exl-churn-key.pem" ec2-user@your-instance-public-dns
```

### 3. Setup Environment on EC2
```bash
# Update system
sudo yum update -y

# Install Python 3 and pip
sudo yum install python3 python3-pip -y

# Install git
sudo yum install git -y

# Clone your repository
git clone https://github.com/yourusername/exl-credit-churn-analysis.git
cd exl-credit-churn-analysis

# Install requirements
pip3 install -r requirements.txt
```

### 4. Run Scripts on EC2
```bash
# Run data cleaning
python3 scripts/data_cleaner.py

# Run feature engineering
python3 scripts/feature_engineering.py

# Run model training
python3 scripts/model_training.py

# Run predictions
python3 scripts/model_predict.py
```

### 5. Transfer Files
```bash
# Copy files to EC2
scp -i "exl-churn-key.pem" local-file.csv ec2-user@your-instance-public-dns:~/

# Copy files from EC2
scp -i "exl-churn-key.pem" ec2-user@your-instance-public-dns:~/remote-file.csv ./
```