# PostgreSQL

## SELECT Query

```sql
SELECT DISTINCT column1, column2, NOW() AS current_time
FROM table_name
WHERE condition1
    AND condition2
    OR condition3
ORDER BY column1 ASC, column2 DESC NULLS LAST -- can also be NULLS FIRST
OFFSET 10 LIMIT 10;
```

## Operators

| Operator | Description |
| --- | --- |
| `=`, `<>`, `!=`, `>`, `<`, `>=`, `<=` | Equal, Not Equal, Greater Than, Less Than, Greater Than or Equal, Less Than or Equal |
| `::` | Cast a value to a different data type |
| `BETWEEN` | Between a range, like `payment_date BETWEEN '2007-02-15' AND '2007-02-20'` |
| `IN` | In a list of values, like `col_name IN ('val1', 'val2')` |
| `LIKE` | Pattern matching, `%` for any number of characters, `_` for a single character |
| `IS NULL`, `IS NOT NULL` | Check if value is NULL or not NULL |
| `AND`, `OR`, `NOT` | Logical operators |
| `ANY`, `ALL` | Compare a value to a set of values, like `col_name > ANY (1, 2, 3)` |
| `EXISTS`, `NOT EXISTS` | Check if a subquery returns any rows |

## JOINs

`INNER JOIN`, `LEFT JOIN` = `LEFT OUTER JOIN`, `RIGHT JOIN` = `RIGHT OUTER JOIN`, `FULL JOIN` = `FULL OUTER JOIN`, `CROSS JOIN`, `NATURAL JOIN` are supported.

## UNION, UNION ALL, INTERSECT, EXCEPT

`UNION` removes duplicates, `UNION ALL` keeps duplicates, `INTERSECT` returns common rows, `EXCEPT` returns rows in first query but not in second query.

## Boolean Operators

- Boolean can have three values: `true`, `false`, and `null`.
- `true` -> true, `t`, `y`, `yes`, `true`, `1`
- `false` -> false, `f`, `n`, `no`, `false`, `0`

## GROUP BY (grouping sets)

```sql
SELECT
	GROUPING(brand) grouping_brand, -- 1 if brand is NULL, 0 otherwise
	GROUPING(segment) grouping_segment,
	brand,
	segment,
	SUM (quantity)
FROM
	sales
GROUP BY
	GROUPING SETS ( -- multiple GROUP BYs, 
        (brand, segment), -- group by brand and segment
		(brand), -- group by brand, segment is NULL
		(segment), -- group by segment, brand is NULL
		() -- no grouping
	)
HAVING GROUPING(brand) = 0 -- only show rows where brand is not NULL
ORDER BY
	brand,
	segment;
```

## GROUP BY (full rollup)

```sql
SELECT segment, brand, SUM (quantity) FROM sales 
GROUP BY 
    ROLLUP (segment, brand) -- full rollup, generates all levels in hierarchy
ORDER BY segment, brand
-- above is equivalent to
SELECT segment, brand, SUM (quantity) FROM sales 
GROUP BY 
    GROUPING SETS (
      (segment, brand),
      (segment),
      ()
    )
ORDER BY segment, brand
```

## GROUP BY (partial rollup)

```sql
SELECT segment, brand, SUM (quantity) FROM sales 
GROUP BY 
    segment, ROLLUP (brand) -- partial rollup
ORDER BY segment, brand
-- above is equivalent to
SELECT segment, brand, SUM (quantity) FROM sales 
GROUP BY 
    GROUPING SETS (
      (segment, brand),
      (segment)
    )
ORDER BY segment, brand
```

## GROUP BY (cube)

```sql
SELECT segment, brand, SUM (quantity) FROM sales
GROUP BY
    CUBE (brand, segment) -- generate all possible combinations
ORDER BY segment, brand
-- above is equivalent to
SELECT segment, brand, SUM (quantity) FROM sales
GROUP BY
    GROUPING SETS (
      (segment, brand),
      (segment),
      (brand),
      ()
    )
ORDER BY segment, brand
```

## ANY and ALL

```sql
SELECT * FROM employees
WHERE salary < ALL (SELECT salary FROM managers)
-- use ANY / ALL with <, >, =, <=, >=, <> and subquery
ORDER BY salary DESC;
```

## EXISTS

```sql
SELECT first_name, last_name FROM customer c
WHERE EXISTS ( -- check if subquery returns any rows
    SELECT 1 FROM payment p
    WHERE p.customer_id = c.customer_id AND amount > 11 -- correlated subquery
)
```

## Common Table Expressions (CTE)

```sql
WITH film_stats AS (
    SELECT AVG(rental_rate) AS avg_rental_rate, MAX(length) AS max_length, MIN(length) AS min_length
    FROM film
), customer_stats AS (
    SELECT COUNT(DISTINCT customer_id) AS total_customers, SUM(amount) AS total_payments
    FROM payment
), main_query AS (
    SELECT
        ROUND((SELECT avg_rental_rate FROM film_stats), 2) AS avg_film_rental_rate,
        (SELECT max_length FROM film_stats) AS max_film_length,
        (SELECT min_length FROM film_stats) AS min_film_length,
        (SELECT total_customers FROM customer_stats) AS total_customers,
        (SELECT total_payments FROM customer_stats) AS total_payments
) SELECT * FROM main_query;
```

## Recursive CTE

```sql