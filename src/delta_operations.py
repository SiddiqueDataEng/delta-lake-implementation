"""
Delta Lake Operations
ACID transactions and time travel for data lakes
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from delta.tables import DeltaTable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeltaLakeManager:
    """Manage Delta Lake operations"""
    
    def __init__(self):
        self.spark = self._create_spark_session()
        
    def _create_spark_session(self):
        """Create Spark session with Delta Lake support"""
        return SparkSession.builder \
            .appName("Delta Lake Operations") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
    
    def create_delta_table(self, df, path, partition_cols=None, mode="overwrite"):
        """Create a new Delta table"""
        logger.info(f"Creating Delta table at {path}")
        
        writer = df.write.format("delta").mode(mode)
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        
        writer.save(path)
        logger.info("Delta table created successfully")
    
    def read_delta_table(self, path, version=None, timestamp=None):
        """Read Delta table with optional time travel"""
        
        reader = self.spark.read.format("delta")
        
        if version is not None:
            logger.info(f"Reading Delta table version {version}")
            reader = reader.option("versionAsOf", version)
        elif timestamp is not None:
            logger.info(f"Reading Delta table as of {timestamp}")
            reader = reader.option("timestampAsOf", timestamp)
        
        return reader.load(path)
    
    def upsert_data(self, delta_path, updates_df, merge_keys):
        """Perform UPSERT operation using MERGE"""
        logger.info("Performing UPSERT operation")
        
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        
        # Build merge condition
        merge_condition = " AND ".join([f"target.{key} = updates.{key}" for key in merge_keys])
        
        # Perform merge
        delta_table.alias("target") \
            .merge(
                updates_df.alias("updates"),
                merge_condition
            ) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()
        
        logger.info("UPSERT completed successfully")
    
    def delete_records(self, delta_path, condition):
        """Delete records matching condition"""
        logger.info(f"Deleting records with condition: {condition}")
        
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        delta_table.delete(condition)
        
        logger.info("Delete operation completed")
    
    def update_records(self, delta_path, condition, updates):
        """Update records matching condition"""
        logger.info(f"Updating records with condition: {condition}")
        
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        delta_table.update(condition, updates)
        
        logger.info("Update operation completed")
    
    def optimize_table(self, delta_path):
        """Optimize Delta table (compaction)"""
        logger.info("Optimizing Delta table")
        
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        delta_table.optimize().executeCompaction()
        
        logger.info("Optimization completed")
    
    def vacuum_table(self, delta_path, retention_hours=168):
        """Remove old files (vacuum)"""
        logger.info(f"Vacuuming Delta table (retention: {retention_hours} hours)")
        
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        delta_table.vacuum(retention_hours)
        
        logger.info("Vacuum completed")
    
    def get_history(self, delta_path, limit=10):
        """Get table history"""
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        return delta_table.history(limit)
    
    def restore_version(self, delta_path, version):
        """Restore table to specific version"""
        logger.info(f"Restoring table to version {version}")
        
        self.spark.sql(f"RESTORE TABLE delta.`{delta_path}` TO VERSION AS OF {version}")
        
        logger.info("Restore completed")
    
    def add_constraint(self, delta_path, constraint_name, condition):
        """Add data quality constraint"""
        logger.info(f"Adding constraint: {constraint_name}")
        
        self.spark.sql(f"""
            ALTER TABLE delta.`{delta_path}`
            ADD CONSTRAINT {constraint_name}
            CHECK ({condition})
        """)
        
        logger.info("Constraint added")
    
    def streaming_write(self, streaming_df, delta_path, checkpoint_path):
        """Write streaming data to Delta table"""
        logger.info("Starting streaming write to Delta")
        
        query = streaming_df \
            .writeStream \
            .format("delta") \
            .outputMode("append") \
            .option("checkpointLocation", checkpoint_path) \
            .start(delta_path)
        
        return query


def example_usage():
    """Example Delta Lake operations"""
    
    manager = DeltaLakeManager()
    
    # Create sample data
    data = [
        (1, "John", "john@email.com", "2020-01-01"),
        (2, "Jane", "jane@email.com", "2020-01-02"),
        (3, "Bob", "bob@email.com", "2020-01-03")
    ]
    df = manager.spark.createDataFrame(data, ["id", "name", "email", "date"])
    
    # Create Delta table
    delta_path = "/tmp/delta/customers"
    manager.create_delta_table(df, delta_path)
    
    # Read table
    customers = manager.read_delta_table(delta_path)
    customers.show()
    
    # Upsert new data
    updates = [
        (2, "Jane Smith", "jane.smith@email.com", "2020-01-04"),
        (4, "Alice", "alice@email.com", "2020-01-04")
    ]
    updates_df = manager.spark.createDataFrame(updates, ["id", "name", "email", "date"])
    manager.upsert_data(delta_path, updates_df, ["id"])
    
    # Read updated table
    customers = manager.read_delta_table(delta_path)
    customers.show()
    
    # Time travel - read previous version
    previous_version = manager.read_delta_table(delta_path, version=0)
    previous_version.show()
    
    # Get history
    history = manager.get_history(delta_path)
    history.show()
    
    # Optimize table
    manager.optimize_table(delta_path)


if __name__ == "__main__":
    example_usage()
