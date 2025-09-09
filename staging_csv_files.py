import os, time, socket, getpass, traceback, glob
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mysql import TEXT

# ====== CONFIGURATION ======
HOST = "localhost"
PORT = 3306
USER = "root"
PASSWORD = "vicky2004"
DATABASE = "adventureworks"
LOAD_MODE = "replace"  # or "append"
PROCESS_NAME = "python_csv_loader"
AUDIT_TABLE = "etl_audit"
CSV_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_audit_table(conn):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{AUDIT_TABLE}` (
        audit_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        process_name VARCHAR(128) NOT NULL,
        database_name VARCHAR(128) NOT NULL,
        table_name VARCHAR(128) NOT NULL,
        load_mode ENUM('replace','append') NOT NULL,
        csv_path TEXT,
        records_attempted BIGINT,
        records_inserted BIGINT,
        started_at DATETIME NOT NULL,
        finished_at DATETIME NOT NULL,
        duration_seconds DECIMAL(12,3) NOT NULL,
        status ENUM('SUCCESS','FAIL') NOT NULL,
        error_message TEXT,
        triggered_by VARCHAR(128),
        host_name VARCHAR(128),
        mysql_user VARCHAR(128)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    conn.execute(text(ddl))

def load_csv_to_mysql(csv_path, table_name, engine):
    df = pd.read_csv(csv_path, encoding="latin-1", dtype=str, na_filter=False)
    attempted = len(df)
    triggered_by = getpass.getuser()
    host_name = socket.gethostname()
    started_ts = time.time()
    mysql_user = None

    try:
        with engine.begin() as conn:
            mysql_user = conn.execute(text("SELECT CURRENT_USER()")).scalar()
            ensure_audit_table(conn)
            dtype_map = {col: TEXT() for col in df.columns}
            df.to_sql(
                table_name,
                conn,
                if_exists=LOAD_MODE,
                index=False,
                dtype=dtype_map,
                method="multi",
                chunksize=1000
            )
            inserted = attempted
            finished_ts = time.time()
            conn.execute(
                text(f"""
                    INSERT INTO `{AUDIT_TABLE}`
                    (process_name, database_name, table_name, load_mode, csv_path,
                     records_attempted, records_inserted, started_at, finished_at,
                     duration_seconds, status, error_message, triggered_by, host_name, mysql_user)
                    VALUES
                    (:process_name, :database_name, :table_name, :load_mode, :csv_path,
                     :records_attempted, :records_inserted, FROM_UNIXTIME(:started), FROM_UNIXTIME(:finished),
                     :duration, 'SUCCESS', NULL, :triggered_by, :host_name, :mysql_user)
                """),
                dict(
                    process_name=PROCESS_NAME,
                    database_name=DATABASE,
                    table_name=table_name,
                    load_mode=LOAD_MODE,
                    csv_path=os.path.abspath(csv_path),
                    records_attempted=attempted,
                    records_inserted=inserted,
                    started=started_ts,
                    finished=finished_ts,
                    duration=round(finished_ts - started_ts, 3),
                    triggered_by=triggered_by,
                    host_name=host_name,
                    mysql_user=mysql_user,
                ),
            )
        print(f"✅ Loaded {attempted} rows into `{DATABASE}.{table_name}`. Audit logged.")
    except Exception as e:
        finished_ts = time.time()
        err = traceback.format_exc(limit=5)
        with engine.begin() as conn:
            ensure_audit_table(conn)
            if mysql_user is None:
                try:
                    mysql_user = conn.execute(text("SELECT CURRENT_USER()")).scalar()
                except Exception:
                    mysql_user = None
            conn.execute(
                text(f"""
                    INSERT INTO `{AUDIT_TABLE}`
                    (process_name, database_name, table_name, load_mode, csv_path,
                     records_attempted, records_inserted, started_at, finished_at,
                     duration_seconds, status, error_message, triggered_by, host_name, mysql_user)
                    VALUES
                    (:process_name, :database_name, :table_name, :load_mode, :csv_path,
                     :records_attempted, 0, FROM_UNIXTIME(:started), FROM_UNIXTIME(:finished),
                     :duration, 'FAIL', :error_message, :triggered_by, :host_name, :mysql_user)
                """),
                dict(
                    process_name=PROCESS_NAME,
                    database_name=DATABASE,
                    table_name=table_name,
                    load_mode=LOAD_MODE,
                    csv_path=os.path.abspath(csv_path),
                    records_attempted=attempted,
                    started=started_ts,
                    finished=finished_ts,
                    duration=round(finished_ts - started_ts, 3),
                    error_message=err[:5000],
                    triggered_by=triggered_by,
                    host_name=host_name,
                    mysql_user=mysql_user,
                ),
            )
        print(f"❌ Load failed for {csv_path}. Error logged.")
        raise

def main():
    url = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}?charset=utf8mb4"
    engine = create_engine(url, pool_pre_ping=True)
    csv_files = glob.glob(os.path.join(CSV_DIR, "AdventureWorks_*.csv"))
    for csv_file in csv_files:
        # Table name: remove prefix and extension, lowercase
        base = os.path.basename(csv_file)
        table_name = base.replace("AdventureWorks_", "").replace(".csv", "").lower()
        load_csv_to_mysql(csv_file, table_name, engine)

if __name__ == "__main__":
    main()
