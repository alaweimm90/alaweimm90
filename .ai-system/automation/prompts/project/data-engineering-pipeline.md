# Data Engineering Pipeline Framework

## Purpose

Design and implement scalable, reliable data processing systems.

---

## 1. INGESTION LAYER

### Batch vs Streaming Decision

```yaml
batch_ingestion:
  use_when:
    - Data freshness > 1 hour acceptable
    - Large volume, periodic updates
    - Complex transformations needed
  tools: [airflow, prefect, dagster]

streaming_ingestion:
  use_when:
    - Real-time requirements
    - Event-driven architecture
    - Continuous data flow
  tools: [kafka, kinesis, pulsar]

hybrid:
  use_when:
    - Both real-time and batch needs
    - Lambda/Kappa architecture
  pattern: streaming_first_with_batch_correction
```

### Data Sources

```yaml
sources:
  databases:
    - type: postgresql
      method: cdc # Change Data Capture
      tool: debezium
    - type: mongodb
      method: change_streams

  apis:
    - type: rest
      pagination: cursor_based
      rate_limiting: respect_headers
    - type: graphql
      batching: enabled

  files:
    - type: s3
      format: parquet
      partitioning: date_based
    - type: sftp
      schedule: hourly
      encryption: pgp

  events:
    - type: kafka
      topics: [events.user, events.order]
      consumer_group: etl_pipeline
```

### Schema Validation

```python
# Pandera or Great Expectations
schema:
  columns:
    user_id:
      dtype: int64
      nullable: false
      unique: true
    email:
      dtype: string
      regex: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
    created_at:
      dtype: datetime64
      nullable: false
    amount:
      dtype: float64
      range: [0, 1000000]
```

### Error Handling

```yaml
error_strategies:
  dead_letter_queue:
    destination: s3://bucket/dlq/
    retention: 30_days
    alerting: enabled

  retry_policy:
    max_attempts: 3
    backoff: exponential
    initial_delay: 1s
    max_delay: 60s

  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60s
```

---

## 2. TRANSFORMATION LAYER

### ETL vs ELT

```yaml
etl:
  description: Transform before loading
  use_when:
    - Sensitive data needs masking
    - Complex business logic
    - Target has limited compute
  tools: [spark, dbt, pandas]

elt:
  description: Load then transform
  use_when:
    - Target has strong compute (warehouse)
    - Schema flexibility needed
    - Faster initial load required
  tools: [dbt, snowflake, bigquery]
```

### Data Quality Checks

```yaml
quality_checks:
  completeness:
    - null_percentage < 5%
    - required_fields_present

  accuracy:
    - referential_integrity
    - business_rule_validation

  consistency:
    - cross_source_matching
    - temporal_consistency

  timeliness:
    - data_freshness < 1_hour
    - processing_sla_met

  uniqueness:
    - primary_key_unique
    - no_duplicates
```

### Deduplication Strategies

```yaml
deduplication:
  exact_match:
    keys: [id, timestamp]
    keep: first # or last, all

  fuzzy_match:
    algorithm: levenshtein
    threshold: 0.85
    fields: [name, address]

  window_based:
    partition_by: user_id
    order_by: timestamp
    keep: latest
```

### Type Conversion

```yaml
type_mappings:
  source_to_target:
    VARCHAR: STRING
    DECIMAL: FLOAT64
    TIMESTAMP: DATETIME
    BOOLEAN: BOOL

  null_handling:
    strings: empty_string # or null
    numbers: 0 # or null
    dates: null
```

---

## 3. STORAGE LAYER

### Data Lake vs Warehouse

```yaml
data_lake:
  use_for:
    - Raw data storage
    - Schema-on-read
    - Diverse data types
  format: parquet, delta, iceberg
  storage: s3, gcs, adls

data_warehouse:
  use_for:
    - Structured analytics
    - SQL queries
    - BI dashboards
  options: [snowflake, bigquery, redshift, databricks]

lakehouse:
  use_for:
    - Best of both worlds
    - ACID on data lake
  format: delta_lake, iceberg, hudi
```

### Partitioning Strategy

```yaml
partitioning:
  time_based:
    column: created_date
    granularity: day # hour, month, year
    format: year=YYYY/month=MM/day=DD

  hash_based:
    column: user_id
    num_buckets: 256

  composite:
    - column: region
      type: list
    - column: date
      type: range
```

### Retention Policies

```yaml
retention:
  hot_tier:
    duration: 30_days
    storage: ssd
    access: frequent

  warm_tier:
    duration: 1_year
    storage: standard
    access: occasional

  cold_tier:
    duration: 7_years
    storage: glacier
    access: rare

  deletion:
    policy: gdpr_compliant
    audit_log: enabled
```

### Access Controls

```yaml
access_control:
  row_level:
    - role: analyst
      filter: region = user_region
    - role: admin
      filter: none

  column_level:
    - column: ssn
      mask: hash
      visible_to: [compliance]
    - column: salary
      mask: null
      visible_to: [hr, finance]

  audit:
    log_queries: true
    log_access: true
    retention: 2_years
```

---

## 4. ORCHESTRATION

### Workflow Scheduler

```yaml
scheduler:
  tool: airflow # or prefect, dagster

  dag_config:
    schedule: '0 2 * * *' # 2 AM daily
    catchup: false
    max_active_runs: 1
    retries: 3
    retry_delay: 5m
```

### Dependencies

```yaml
dependencies:
  task_dependencies: extract >> transform >> load >> validate

  external_dependencies:
    - type: sensor
      target: s3://bucket/data/
      timeout: 2h
    - type: api
      endpoint: /health
      expected: 200

  cross_dag:
    - dag: upstream_dag
      task: final_task
      mode: wait
```

### Retry Logic

```yaml
retry_config:
  exponential_backoff:
    initial: 30s
    multiplier: 2
    max: 1h

  conditional_retry:
    - error: connection_timeout
      retries: 5
    - error: data_validation
      retries: 0 # fail fast
    - error: rate_limit
      retries: 10
      delay: 60s
```

### Monitoring and Alerts

```yaml
monitoring:
  metrics:
    - pipeline_duration
    - records_processed
    - error_count
    - data_freshness

  dashboards:
    tool: grafana
    panels:
      - pipeline_status
      - throughput_chart
      - error_timeline

  alerts:
    - condition: pipeline_failed
      channel: pagerduty
      severity: critical
    - condition: duration > 2x_average
      channel: slack
      severity: warning
    - condition: data_quality_score < 95%
      channel: email
      severity: warning
```

---

## Pipeline Template

```python
# Airflow DAG example
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
}

with DAG(
    'data_pipeline',
    default_args=default_args,
    schedule_interval='0 2 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_data,
    )

    validate_source = PythonOperator(
        task_id='validate_source',
        python_callable=validate_source_data,
    )

    transform = PythonOperator(
        task_id='transform',
        python_callable=transform_data,
    )

    load = PythonOperator(
        task_id='load',
        python_callable=load_to_warehouse,
    )

    validate_target = PythonOperator(
        task_id='validate_target',
        python_callable=validate_loaded_data,
    )

    extract >> validate_source >> transform >> load >> validate_target
```

---

## Checklist

- [ ] Data sources identified and documented
- [ ] Schema validation implemented
- [ ] Error handling configured
- [ ] Transformations tested
- [ ] Data quality checks in place
- [ ] Partitioning strategy defined
- [ ] Retention policies configured
- [ ] Access controls implemented
- [ ] Orchestration scheduled
- [ ] Monitoring and alerts active
- [ ] Documentation complete
