import mysql.connector
import pandas as pd
import os
from datetime import datetime


def create_database_and_table(host='localhost', user='root', password='', database='credit_churn_db'):

    try:
        # Connect to MySQL server (without database)
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        cursor = connection.cursor()
        
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        print(f" Database '{database}' created or already exists")
        
        cursor.close()
        connection.close()
        
        # Connect to the specific database
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        cursor = connection.cursor()
        
        # Create table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS customers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            CustomerID VARCHAR(20) NOT NULL UNIQUE,
            Gender VARCHAR(10) NOT NULL,
            Age DECIMAL(5,2),
            Tenure DECIMAL(5,2),
            Balance DECIMAL(15,2),
            NumOfProducts DECIMAL(5,2),
            HasCrCard TINYINT(1),
            IsActiveMember DECIMAL(5,2),
            EstimatedSalary DECIMAL(15,2),
            Churn DECIMAL(5,2)
        )
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        print(" Table 'customers' created successfully")
        
        cursor.close()
        return connection
        
    except mysql.connector.Error as err:
        print(f" Error setting up database: {err}")
        return None

def load_csv_to_mysql(csv_file_path, connection, table_name='customers'):
    try:
        
        df = pd.read_csv(csv_file_path)
        print(f"Loaded CSV file with {len(df)} rows")
        
        cursor = connection.cursor()
        # Insert data
        insert_query = f"""
        INSERT INTO {table_name} 
        (CustomerID, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Churn)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        rows_inserted = 0
        errors = 0
        
        for index, row in df.iterrows():
            try:
                cursor.execute(insert_query, (
                    str(row['CustomerID']),
                    str(row['Gender']),
                    float(row['Age']) if pd.notna(row['Age']) else None,
                    float(row['Tenure']) if pd.notna(row['Tenure']) else None,
                    float(row['Balance']) if pd.notna(row['Balance']) else None,
                    float(row['NumOfProducts']) if pd.notna(row['NumOfProducts']) else None,
                    int(row['HasCrCard']) if pd.notna(row['HasCrCard']) else 0,
                    float(row['IsActiveMember']) if pd.notna(row['IsActiveMember']) else None,
                    float(row['EstimatedSalary']) if pd.notna(row['EstimatedSalary']) else None,
                    float(row['Churn']) if pd.notna(row['Churn']) else None
                ))
                rows_inserted += 1
                
                # Commit every 100
                if rows_inserted % 100 == 0:
                    connection.commit()
                    print(f"  Inserted {rows_inserted} rows...")
                    
            except mysql.connector.Error as err:
                errors += 1
                print(f"  Warning: Error inserting row {index}: {err}")
                
        # Final commit
        connection.commit()
        cursor.close()
        
        print(f" Successfully inserted {rows_inserted} rows")
        if errors > 0:
            print(f"{errors} rows had errors and were skipped")
            
        return rows_inserted
        
    except Exception as err:
        print(f" Error loading CSV to MySQL: {err}")
        return 0

def export_mysql_to_csv(connection, table_name='customers', output_file_path=None):
    """Export data from MySQL table to CSV file"""
    try:
        cursor = connection.cursor()
        
        # Get all data from table
        select_query = f"SELECT * FROM {table_name}"
        cursor.execute(select_query)
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=column_names)
        
        print(f"✓ Retrieved {len(df)} rows from MySQL table '{table_name}'")
        
        # Set default output path if not provided
        if output_file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file_path = os.path.join(base_dir, 'data', 'exports', f'exported_data_{timestamp}.csv')
        
        # Create exports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file_path, index=False)
        
        print(f"✓ Data exported successfully to: {output_file_path}")
        print(f"✓ Export contains {len(df)} rows and {len(df.columns)} columns")
        
        # Show first few rows as preview
        print("\n📊 Data Preview:")
        print(df.head().to_string())
        
        cursor.close()
        return output_file_path, len(df)
        
    except mysql.connector.Error as err:
        print(f"❌ MySQL Error during export: {err}")
        return None, 0
    except Exception as err:
        print(f"❌ Error exporting data to CSV: {err}")
        return None, 0

def get_table_info(connection, table_name='customers'):
    """Get information about the table structure and data"""
    try:
        cursor = connection.cursor()
        
        # Get table structure
        cursor.execute(f"DESCRIBE {table_name}")
        structure = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        print(f"\n📋 Table '{table_name}' Information:")
        print(f"  Total rows: {row_count}")
        print(f"  Table structure:")
        
        for column in structure:
            print(f"    {column[0]}: {column[1]} {'(Primary Key)' if column[3] == 'PRI' else ''}")
        
        cursor.close()
        return row_count, structure
        
    except mysql.connector.Error as err:
        print(f"❌ Error getting table info: {err}")
        return 0, []

def main():
    print("Credit Churn Analysis - CSV to MySQL Loader")
    print("=" * 50)
    
    # Database configuration
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'root',
        'password': 'Lokesh@8978',  # Update with your MySQL password
        'database': 'exl_churn_db'
    }
    
    # CSV file path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file_path = os.path.join(base_dir, 'data', 'processed', 'churn_cleaned.csv')
    
    print(f"CSV file path: {csv_file_path}")
    
    try:
        # Create database and table
        connection = create_database_and_table(**DB_CONFIG)
        if not connection:
            return False
        
        # Load CSV data
        rows_inserted = load_csv_to_mysql(csv_file_path, connection)
        
        if rows_inserted > 0:
            
            # Show connection info
            print(f"\nDatabase Details:")
            print(f"  Host: {DB_CONFIG['host']}")
            print(f"  Database: {DB_CONFIG['database']}")
            print(f"  Table: customer_churn")
            print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        return True
        
    except Exception as e:
        print(f" Error in main process: {e}")
        return False
        
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Database connection closed")

if __name__ == "__main__":
    main()