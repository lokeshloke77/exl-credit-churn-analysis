# AWS RDS Setup Guide for Aurora MySQL

## Step-by-step AWS RDS Setup Guide (Aurora MySQL)

### 1. Create Aurora MySQL Cluster
```bash
aws rds create-db-cluster \
    --db-cluster-identifier exl-churn-cluster \
    --engine aurora-mysql \
    --master-username admin \
    --master-user-password YourPassword123 \
    --vpc-security-group-ids sg-xxxxxxxx \
    --db-subnet-group-name default
```

### 2. Create Aurora MySQL Instance
```bash
aws rds create-db-instance \
    --db-instance-identifier exl-churn-instance \
    --db-instance-class db.r5.large \
    --engine aurora-mysql \
    --db-cluster-identifier exl-churn-cluster
```

### 3. Configure Security Groups
- Allow inbound traffic on port 3306
- Restrict access to your IP address
- Configure VPC and subnet groups

### 4. Connection String
```python
import mysql.connector

config = {
    'host': 'exl-churn-cluster.cluster-xxxxxxxx.region.rds.amazonaws.com',
    'port': 3306,
    'user': 'admin',
    'password': 'YourPassword123',
    'database': 'churn_db'
}

connection = mysql.connector.connect(**config)
```

### 5. Data Migration
```sql
CREATE DATABASE churn_db;
USE churn_db;

CREATE TABLE customer_data (
    customer_id INT PRIMARY KEY,
    age INT,
    tenure INT,
    balance DECIMAL(10,2),
    num_of_products INT,
    has_cr_card TINYINT,
    is_active_member TINYINT,
    estimated_salary DECIMAL(10,2),
    churn TINYINT
);
```