# dbt (data build tool)

## What is dbt and how does it differ from other tools?

dbt is a transformation tool that lets you:
- Write data transformations in SQL/Python
- Version control your data transformations
- Test your data quality
- Document your data models
- Create dependencies between models
- Deploy to multiple environments

Key differences from other tools:
- SQL-first approach vs traditional ETL tools
- Version control integration
- Built-in testing framework
- Modular model approach
- Focus on transformation (T in ELT)
- Strong documentation capabilities

## How do you structure a dbt project?

Basic dbt project structure:
```plaintext
my_dbt_project/
├── dbt_project.yml          # Project configuration
├── packages.yml             # External package dependencies
├── profiles.yml            # Connection configurations
├── models/                 # SQL model definitions
│   ├── staging/           # Raw data models
│   ├── intermediate/      # Business logic layers
│   └── marts/            # Final presentation layer
├── tests/                 # Custom test definitions
├── snapshots/            # SCD Type 2 definitions
├── macros/              # Reusable SQL snippets
├── seeds/               # Static CSV files
└── docs/               # Documentation files
```

## How do you write models in dbt?

Basic model structure:
```sql
-- models/marts/core/dim_customers.sql

{{ config(
    materialized='table',
    schema='core',
    tags=['daily', 'customers']
) }}

WITH customer_orders AS (
    SELECT 
        customer_id,
        COUNT(*) as order_count,
        SUM(amount) as total_amount
    FROM {{ ref('stg_orders') }}
    GROUP BY customer_id
),

final AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        co.order_count,
        co.total_amount
    FROM {{ ref('stg_customers') }} c
    LEFT JOIN customer_orders co
        ON c.customer_id = co.customer_id
)

SELECT * FROM final
```

## What are the different materialization types?

Common materialization types:
```yaml
# models/example/my_model.sql
{{ config(materialized='table') }}  # Creates a table
{{ config(materialized='view') }}   # Creates a view
{{ config(materialized='incremental') }}  # Incrementally updated table
{{ config(materialized='ephemeral') }}    # CTEs for other models

# Project-level configuration in dbt_project.yml
models:
  my_project:
    staging:
      materialized: view
    marts:
      materialized: table
```

## How do you implement testing in dbt?

Data testing approaches:
```yaml
# models/schema.yml
version: 2

models:
  - name: dim_customers
    columns:
      - name: customer_id
        tests:
          - unique
          - not_null
      - name: email
        tests:
          - unique
          - not_null
          - accepted_values:
              values: ['gmail.com', 'yahoo.com']
              quote: false

# Custom test definition
# tests/assert_positive_values.sql
{% test assert_positive_values(model, column_name) %}
SELECT *
FROM {{ model }}
WHERE {{ column_name }} < 0
{% endtest %}
```

## How do you document your models?

Documentation examples:
```yaml
# models/schema.yml
version: 2

models:
  - name: dim_customers
    description: "Core customer dimension table"
    columns:
      - name: customer_id
        description: "Primary key of the customers table"
        tests:
          - unique
          - not_null
      - name: first_name
        description: "Customer's first name"
      - name: last_name
        description: "Customer's last name"

# Create documentation blocks
{% docs customer_status %}
Customer status can be one of the following values:
* active - Customer has made a purchase in the last 90 days
* inactive - Customer hasn't made a purchase in 90+ days
* churned - Customer hasn't made a purchase in 365+ days
{% enddocs %}
```

## How do you implement macros?

Macro examples:
```sql
-- macros/generate_schema_name.sql
{% macro generate_schema_name(custom_schema_name, node) %}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ default_schema }}_{{ custom_schema_name }}
    {%- endif -%}
{% endmacro %}

-- macros/clean_stale_models.sql
{% macro clean_stale_models(database=target.database, schema=target.schema, days=7, dry_run=True) %}
    {% set get_drop_commands_query %}
        SELECT
            'DROP TABLE {{ database }}.{{ schema }}.' || table_name || ';'
        FROM information_schema.tables
        WHERE table_schema = '{{ schema }}'
        AND last_altered < CURRENT_DATE - {{ days }}
    {% endset %}
    
    {{ log('\nGenerating cleanup queries...\n', info=True) }}
    {% set drop_queries = run_query(get_drop_commands_query).columns[0].values() %}
    
    {% for query in drop_queries %}
        {% if dry_run %}
            {{ log(query, info=True) }}
        {% else %}
            {{ run_query(query) }}
        {% endif %}
    {% endfor %}
{% endmacro %}
```

## How do you handle incremental models?

Incremental model implementation:
```sql
-- models/marts/fact_orders.sql
{{ config(
    materialized='incremental',
    unique_key='order_id',
    incremental_strategy='merge'
) }}

SELECT 
    order_id,
    customer_id,
    order_date,
    amount,
    status
FROM {{ ref('stg_orders') }}

{% if is_incremental() %}
    WHERE order_date > (
        SELECT MAX(order_date)
        FROM {{ this }}
    )
{% endif %}
```

## How do you implement snapshots?

Snapshot implementation:
```sql
-- snapshots/order_status_history.sql
{% snapshot order_status_snapshot %}

{{
    config(
      target_schema='snapshots',
      unique_key='order_id',
      strategy='timestamp',
      updated_at='updated_at',
    )
}}

SELECT 
    order_id,
    status,
    updated_at
FROM {{ source('raw', 'orders') }}

{% endsnapshot %}
```

## How do you manage sources?

Source configuration:
```yaml
# models/schema.yml
version: 2

sources:
  - name: raw
    database: raw_data
    schema: public
    tables:
      - name: customers
        loaded_at_field: _etl_loaded_at
        freshness:
          warn_after: {count: 12, period: hour}
          error_after: {count: 24, period: hour}
        columns:
          - name: id
            tests:
              - unique
              - not_null
      - name: orders
        loaded_at_field: _etl_loaded_at
        freshness:
          warn_after: {count: 6, period: hour}
          error_after: {count: 12, period: hour}
```

## How do you handle seeds?

Seed configuration and usage:
```yaml
# dbt_project.yml
seeds:
  my_project:
    country_codes:
      +column_types:
        country_code: varchar(2)
        country_name: varchar(100)
    exchange_rates:
      +quote_columns: true

# Usage in models
SELECT 
    o.*,
    c.country_name
FROM {{ ref('stg_orders') }} o
LEFT JOIN {{ ref('country_codes') }} c
    ON o.country_code = c.country_code
```

## How do you implement hooks?

Hook implementation:
```sql
-- hooks for specific models
{{ config(
    post_hook=[
        "GRANT SELECT ON {{ this }} TO ROLE_ANALYST",
        "ANALYZE {{ this }}"
    ]
) }}

-- Project level hooks in dbt_project.yml
on-run-start:
  - "SET TIME_ZONE = 'UTC'"
  - "{{ log('Run started at ' ~ modules.datetime.datetime.now(), info=True)}}"

on-run-end:
  - "GRANT USAGE ON SCHEMA {{ target.schema }} TO ROLE_ANALYST"
  - "{{ log('Run completed at ' ~ modules.datetime.datetime.now(), info=True)}}"
```

## How do you handle packages?

Package management:
```yaml
# packages.yml
packages:
  - package: dbt-labs/dbt_utils
    version: 0.8.0
  - package: calogica/dbt_date
    version: 0.5.0
  - git: "https://github.com/org/dbt-package.git"
    revision: v1.0.0

# Usage in models
SELECT 
    order_id,
    {{ dbt_utils.generate_surrogate_key(['customer_id', 'order_date']) }} as order_sk,
    customer_id,
    order_date
FROM {{ ref('stg_orders') }}
```

## How do you implement exposures?

Exposure configuration:
```yaml
# models/schema.yml
version: 2

exposures:
  - name: weekly_sales_dashboard
    type: dashboard
    maturity: high
    url: https://bi-tool.company.com/dashboards/12345
    description: >
      Weekly sales dashboard showing key metrics and trends
    depends_on:
      - ref('fct_daily_sales')
      - ref('dim_customers')
    owner:
      name: Analytics Team
      email: analytics@company.com
```