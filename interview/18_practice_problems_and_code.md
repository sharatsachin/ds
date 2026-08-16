# 18. Hands-On Practice Problems & Runnable Code

This file is different from the rest of the kit. Files 01–14, 16, 17, and 19 **teach** the
material. This file makes you **do** it. Every problem below is something you should
actually attempt — on paper, in a scratch SQL shell, or in a Python REPL — before you look
at the answer.

**Rule of the deck: struggle first, then check.** Every solution lives inside a collapsed
`<details>` block. Read the problem, commit to an answer (write it down, don't just think
"I could probably do that"), then expand the block. If you're wrong or you froze, that's
useful signal about what to re-drill — treat it as a cue to go back to the relevant
companion file, not as a reason to feel bad.

**What you need installed:** Python 3 with `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`shap`, and `matplotlib`; and a SQL engine — `sqlite3` (built into Python, zero setup),
`duckdb`, or `postgres` all work. Every code block in this file is copy-paste runnable as-is.

**A portability note on the SQL in Part 1:** the solutions are written in SQLite-flavored
SQL (the easiest zero-install path — `python3 -m sqlite3` or `sqlite3 :memory:`). Three
things differ if you paste into DuckDB or Postgres instead:
1. **`strftime` argument order.** SQLite: `strftime('%Y-%m', order_date)` (format first).
   DuckDB: `strftime(order_date, '%Y-%m')` (value first, like Python's `.strftime()`).
   Postgres doesn't have `strftime` at all — use `to_char(order_date, 'YYYY-MM')`.
2. **Date arithmetic.** SQLite: `DATE(d, '+1 day')`. DuckDB/Postgres: `d + INTERVAL '1 day'`.
3. **Recursive CTEs** (`WITH RECURSIVE`) work in all three, but DuckDB and Postgres also
   support `generate_series(start_date, end_date, INTERVAL '1 day')` as a cleaner one-line
   alternative to build a date spine — feel free to swap it in.

Everything else (window functions, CTEs, joins, `CASE`) is standard ANSI SQL and runs
unmodified on all three engines.

## Table of Contents
- [Part 1 — SQL Practice Problems](#part-1--sql-practice-problems)
- [Part 2 — Probability & Statistics Practice Problems](#part-2--probability--statistics-practice-problems)
- [Part 3 — "Derive It From Scratch" Prompts](#part-3--derive-it-from-scratch-prompts)
- [Part 4 — Runnable Code Exercises](#part-4--runnable-code-exercises)
- [Part 5 — ML/Coding Whiteboard Questions](#part-5--mlcoding-whiteboard-questions)

---

## Part 1 — SQL Practice Problems

### Sample schema and data

Four tables: customers place orders, orders contain line items, line items reference
products. Paste the block below into `sqlite3` (or DuckDB/Postgres, with the syntax
tweaks noted above) to get a real, queryable dataset.

```sql
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    signup_date  DATE NOT NULL,
    region       TEXT NOT NULL
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL,   -- logical FK -> customers.customer_id
    order_date   DATE NOT NULL,
    amount       NUMERIC NOT NULL,
    status       TEXT NOT NULL       -- 'completed' or 'cancelled'
);

CREATE TABLE order_items (
    order_id    INTEGER NOT NULL,    -- logical FK -> orders.order_id
    product_id  INTEGER NOT NULL,    -- logical FK -> products.product_id
    quantity    INTEGER NOT NULL,
    unit_price  NUMERIC NOT NULL
);

INSERT INTO customers (customer_id, name, signup_date, region) VALUES
(1, 'Alice', '2023-01-05', 'East'),
(2, 'Bob',   '2023-02-10', 'West'),
(3, 'Carol', '2023-01-20', 'East'),
(4, 'Dave',  '2023-03-01', 'West'),
(5, 'Erin',  '2023-04-15', 'North');

INSERT INTO products (product_id, product_name, category) VALUES
(1, 'Widget A', 'Widgets'),
(2, 'Widget B', 'Widgets'),
(3, 'Gadget A', 'Gadgets'),
(4, 'Gadget B', 'Gadgets'),
(5, 'Gizmo A',  'Gizmos');

INSERT INTO orders (order_id, customer_id, order_date, amount, status) VALUES
(1001, 1, '2024-01-03', 120.00, 'completed'),
(1002, 3, '2024-01-05', 75.50,  'completed'),
(1003, 2, '2024-01-10', 200.00, 'completed'),
(1004, 1, '2024-01-15', 60.00,  'completed'),
(1005, 3, '2024-01-22', 130.00, 'completed'),
(1006, 1, '2024-01-28', 90.00,  'cancelled'),
(1007, 1, '2024-02-02', 150.00, 'completed'),
(1008, 4, '2024-02-05', 220.00, 'completed'),
(1009, 3, '2024-02-08', 45.00,  'completed'),
(1010, 1, '2024-02-14', 80.00,  'completed'),
(1011, 4, '2024-02-20', 60.00,  'completed'),
(1012, 1, '2024-03-01', 110.00, 'completed'),
(1013, 2, '2024-03-04', 175.00, 'completed'),
(1014, 4, '2024-03-10', 95.00,  'completed'),
(1015, 1, '2024-03-18', 130.00, 'completed'),
(1016, 2, '2024-03-25', 60.00,  'cancelled'),
(1017, 1, '2024-04-02', 145.00, 'completed'),
(1018, 2, '2024-04-06', 210.00, 'completed'),
(1019, 5, '2024-04-10', 300.00, 'completed'),
(1020, 1, '2024-04-20', 70.00,  'completed'),
(1021, 1, '2024-05-03', 160.00, 'completed'),
(1022, 4, '2024-05-08', 130.00, 'completed'),
(1023, 5, '2024-05-12', 250.00, 'completed'),
(1024, 1, '2024-05-22', 90.00,  'completed'),
(1025, 1, '2024-06-01', 175.00, 'completed'),
(1026, 4, '2024-06-05', 140.00, 'completed'),
(1027, 5, '2024-06-15', 275.00, 'completed'),
(1028, 1, '2024-06-20', 100.00, 'completed');

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1001, 1, 3, 20.00), (1001, 3, 1, 40.00),
(1002, 2, 2, 25.00),
(1003, 4, 2, 55.00), (1003, 5, 1, 90.00),
(1004, 1, 2, 20.00),
(1005, 3, 2, 40.00), (1005, 1, 1, 20.00),
(1006, 2, 3, 25.00),
(1007, 5, 1, 90.00), (1007, 1, 1, 20.00),
(1008, 4, 3, 55.00),
(1009, 2, 1, 25.00),
(1010, 1, 4, 20.00),
(1011, 3, 1, 40.00),
(1012, 5, 1, 90.00),
(1013, 4, 2, 55.00), (1013, 2, 1, 25.00),
(1014, 5, 1, 90.00),
(1015, 1, 3, 20.00), (1015, 3, 1, 40.00),
(1016, 4, 1, 55.00),
(1017, 2, 2, 25.00),
(1018, 5, 2, 90.00),
(1019, 1, 5, 20.00), (1019, 4, 2, 55.00),
(1020, 3, 1, 40.00),
(1021, 5, 1, 90.00),
(1022, 4, 2, 55.00),
(1023, 5, 2, 90.00),
(1024, 1, 2, 20.00),
(1025, 2, 3, 25.00),
(1026, 3, 2, 40.00),
(1027, 1, 3, 20.00),
(1028, 4, 1, 55.00);
```

**Worth noticing about the data before you start:** Alice orders every single month
Jan–Jun (no gaps — a loyal customer, useful as a "should NOT get flagged" control case).
Carol orders in Jan and Feb only, then stops entirely (true churn). Bob orders Jan, Mar,
Apr — missing Feb (a gap he later returns from). Dave orders Feb, Mar, May, Jun — missing
Apr. Erin is a newer customer who only appears starting April. Orders 1006 and 1016 are
`cancelled`. This is deliberate — several problems below are designed so these specific
customers are the "interesting" rows in the answer.

### Problem 1 (warm-up): List all completed orders with the customer's name and region.

<details>
<summary>Show solution</summary>

```sql
SELECT o.order_id, c.name, c.region, o.order_date, o.amount
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
ORDER BY o.order_date;
```

</details>

### Problem 2: Find each customer's first order date (regardless of status).

<details>
<summary>Show solution</summary>

```sql
SELECT customer_id, MIN(order_date) AS first_order_date
FROM orders
GROUP BY customer_id;
```

</details>

### Problem 3: Total revenue (from line items) by region, completed orders only.

<details>
<summary>Show solution</summary>

```sql
SELECT c.region, SUM(oi.quantity * oi.unit_price) AS total_revenue
FROM order_items oi
JOIN orders o    ON o.order_id = oi.order_id
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.region
ORDER BY total_revenue DESC;
```

Note this joins through `order_items`, not `orders.amount` — the line items are the
source of truth for product/revenue-attribution questions in this schema.

</details>

### Problem 4: Find each customer's *second* completed order (by date).

<details>
<summary>Show solution</summary>

```sql
WITH ranked_orders AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY customer_id ORDER BY order_date, order_id
           ) AS rn
    FROM orders
    WHERE status = 'completed'
)
SELECT customer_id, order_id, order_date, amount
FROM ranked_orders
WHERE rn = 2;
```

`ROW_NUMBER()` (not `RANK()`) is the right tool here — ties would otherwise both get rank
1 and you could end up with zero or two rows tagged "second," which is wrong for "the
nth order" style questions. The secondary `ORDER BY ... order_id` breaks ties deterministically
if two orders somehow land on the same date.

</details>

### Problem 5: Compute a running (cumulative) total of revenue per customer, ordered by date.

<details>
<summary>Show solution</summary>

```sql
SELECT customer_id, order_date, amount,
       SUM(amount) OVER (
           PARTITION BY customer_id
           ORDER BY order_date
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running_total
FROM orders
WHERE status = 'completed'
ORDER BY customer_id, order_date;
```

The explicit frame (`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`) is technically the
default for `ORDER BY` + no frame clause, but spelling it out is good practice — it's the
detail that trips people up when they later need a *different* frame (e.g. a trailing
7-row window) and don't realize the default silently changed on them.

</details>

### Problem 6: Find customers whose average completed-order amount is above the overall average.

<details>
<summary>Show solution</summary>

```sql
SELECT customer_id, AVG(amount) AS avg_order_amount
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
HAVING AVG(amount) > (
    SELECT AVG(amount) FROM orders WHERE status = 'completed'
);
```

`HAVING` filters on the aggregate itself, which is why this can't be a `WHERE` clause —
`WHERE` runs before grouping/aggregation happens.

</details>

### Problem 7: Find the top 3 products by revenue within each region.

<details>
<summary>Show solution</summary>

```sql
WITH product_region_revenue AS (
    SELECT c.region,
           p.product_name,
           SUM(oi.quantity * oi.unit_price) AS revenue
    FROM order_items oi
    JOIN orders o    ON o.order_id = oi.order_id
    JOIN customers c ON c.customer_id = o.customer_id
    JOIN products p  ON p.product_id = oi.product_id
    WHERE o.status = 'completed'
    GROUP BY c.region, p.product_name
),
ranked AS (
    SELECT *,
           RANK() OVER (PARTITION BY region ORDER BY revenue DESC) AS rnk
    FROM product_region_revenue
)
SELECT region, product_name, revenue, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY region, rnk;
```

`RANK()` (not `ROW_NUMBER()`) here on purpose: if two products in the same region tie
exactly on revenue, you'd typically want both to show up as "tied for #2," not have one
arbitrarily bumped to #3.

</details>

### Problem 8: Compute month-over-month revenue growth % (completed orders).

<details>
<summary>Show solution</summary>

```sql
WITH monthly_revenue AS (
    SELECT strftime('%Y-%m', order_date) AS month,
           SUM(amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY month
)
SELECT month,
       revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
       ROUND(
           100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
           / LAG(revenue) OVER (ORDER BY month),
           2
       ) AS mom_growth_pct
FROM monthly_revenue
ORDER BY month;
```

(Postgres/DuckDB: swap `strftime('%Y-%m', order_date)` for `to_char(order_date,'YYYY-MM')`
/ `strftime(order_date, '%Y-%m')` respectively, per the portability note above.)

</details>

### Problem 9 (churn-style): Find every customer-month where they ordered in month M but did *not* order in month M+1.

<details>
<summary>Show solution</summary>

```sql
WITH customer_months AS (
    SELECT DISTINCT customer_id,
           CAST(strftime('%Y', order_date) AS INTEGER) * 12
               + CAST(strftime('%m', order_date) AS INTEGER) AS month_idx
    FROM orders
    WHERE status = 'completed'
),
bounds AS (
    SELECT MAX(month_idx) AS max_month_idx FROM customer_months
),
with_next AS (
    SELECT customer_id, month_idx,
           LEAD(month_idx) OVER (PARTITION BY customer_id ORDER BY month_idx) AS next_month_idx
    FROM customer_months
)
SELECT w.customer_id, w.month_idx AS last_active_month_idx
FROM with_next w, bounds b
WHERE w.month_idx < b.max_month_idx                          -- exclude the dataset's final month:
  AND (w.next_month_idx IS NULL                               -- we have no data yet to know if
       OR w.next_month_idx > w.month_idx + 1);                -- month+1 would've had an order
```

Encoding months as `year*12 + month` turns "is the next calendar month missing" into
plain integer arithmetic (`next_month_idx > month_idx + 1`), which is much less error-prone
than trying to compare date strings across year boundaries. The `bounds` CTE matters: without
it, every customer's *most recent* month would trivially show up as "no next month" purely
because the dataset ends there, which is a data-boundary artifact, not real churn.

Against this dataset: **Carol flags at Feb** (true churn — she never orders again), **Bob
flags at Jan** (temporary lapse — he returns in March), **Dave flags at Mar** (temporary
lapse — he returns in May). Alice, and Erin's June, are correctly *not* flagged.

</details>

### Problem 10 (gaps-and-islands, detection): For each customer, find any pair of consecutive order-months with a gap greater than one month between them.

<details>
<summary>Show solution</summary>

```sql
WITH customer_months AS (
    SELECT DISTINCT customer_id,
           CAST(strftime('%Y', order_date) AS INTEGER) * 12
               + CAST(strftime('%m', order_date) AS INTEGER) AS month_idx
    FROM orders
    WHERE status = 'completed'
),
with_prev AS (
    SELECT customer_id, month_idx,
           LAG(month_idx) OVER (PARTITION BY customer_id ORDER BY month_idx) AS prev_month_idx
    FROM customer_months
)
SELECT customer_id,
       prev_month_idx,
       month_idx,
       (month_idx - prev_month_idx - 1) AS months_skipped
FROM with_prev
WHERE month_idx - prev_month_idx > 1;
```

This is deliberately the mirror image of Problem 9: that one used `LEAD` to look *forward*
from a known order and ask "does the customer ever come back," which is a churn/retention
question. This one uses `LAG` to look *backward between two known orders* and ask "was
there silence in between," which is a pure gap-detection question — it can never flag a
customer's final month (there's no "next" order to bound the gap), which is exactly why
Carol's stop-and-never-return doesn't show up here even though she's flagged in Problem 9.
Against this data: **Bob** shows a 1-month gap (Feb skipped, between Jan and Mar), **Dave**
shows a 1-month gap (Apr skipped, between Mar and May).

</details>

### Problem 11 (gaps-and-islands, full form): For each customer, group their order-months into maximal streaks of consecutive months ("islands"), returning each streak's start and end month.

<details>
<summary>Show solution</summary>

```sql
WITH customer_months AS (
    SELECT DISTINCT
           customer_id,
           CAST(strftime('%Y', order_date) AS INTEGER) * 12
               + CAST(strftime('%m', order_date) AS INTEGER) AS month_idx,
           strftime('%Y-%m', order_date) AS month_label
    FROM orders
    WHERE status = 'completed'
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY month_idx) AS rn
    FROM customer_months
),
islands AS (
    -- The classic trick: for a run of consecutive months, (month_idx - rn) is CONSTANT.
    -- Any break in consecutiveness shifts month_idx but not rn, so the difference changes
    -- and a new island starts.
    SELECT *, (month_idx - rn) AS island_key
    FROM ranked
)
SELECT customer_id,
       MIN(month_label) AS streak_start_month,
       MAX(month_label) AS streak_end_month,
       COUNT(*)         AS months_in_streak
FROM islands
GROUP BY customer_id, island_key
ORDER BY customer_id, streak_start_month;
```

Walk through why `month_idx - rn` is constant within an island: if a customer's ordered
months are `[10, 11, 12]` (consecutive), `rn` is `[1, 2, 3]`, so `month_idx - rn` is
`[9, 9, 9]` — constant. If instead the months are `[10, 12]` (a gap at 11), `rn` is still
`[1, 2]`, so `month_idx - rn` is `[9, 10]` — it jumps, which is exactly the signal that a
new island started. Grouping by that constant collapses each island into one row. Run this
against the sample data and confirm: Alice gets one island (Jan–Jun), Bob gets two islands
(Jan–Jan, Mar–Apr), Carol gets one island (Jan–Feb), Dave gets two islands (Feb–Mar,
May–Jun), Erin gets one island (Apr–Jun).

</details>

### Problem 12 (hardest): Compute a 7-day rolling average of total daily revenue across all customers, including days with zero completed orders.

<details>
<summary>Show solution</summary>

```sql
WITH RECURSIVE date_spine(d) AS (
    SELECT DATE('2024-01-01')
    UNION ALL
    SELECT DATE(d, '+1 day') FROM date_spine WHERE d < DATE('2024-06-30')
),
daily_revenue AS (
    SELECT order_date AS d, SUM(amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY order_date
)
SELECT ds.d AS day,
       COALESCE(dr.revenue, 0) AS daily_revenue,
       AVG(COALESCE(dr.revenue, 0)) OVER (
           ORDER BY ds.d
           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS rolling_7day_avg_revenue
FROM date_spine ds
LEFT JOIN daily_revenue dr ON dr.d = ds.d
ORDER BY ds.d;
```

The `date_spine` recursive CTE is the crux of this problem: if you just `GROUP BY
order_date` and slap a rolling window on it, you silently skip every day with zero
completed orders, which shrinks the "7-day" window down to "the last 7 days that
happened to have an order" — wrong. Building an explicit calendar spine and
`LEFT JOIN`-ing revenue onto it (defaulting missing days to 0 via `COALESCE`) is the fix,
and it's the same fix you need for daily active users, daily churn rate, or any other
metric where "no activity" is a real, countable data point, not a missing row.
(DuckDB/Postgres: replace `DATE(d, '+1 day')` with `d + INTERVAL '1 day'`, or just use
`generate_series('2024-01-01'::date, '2024-06-30'::date, INTERVAL '1 day')` instead of the
recursive CTE entirely.)

</details>

---

## Part 2 — Probability & Statistics Practice Problems

### Problem 1 (Bayes' theorem): A company sources widgets from 3 suppliers. Supplier A provides 50% of stock with a 2% defect rate. Supplier B provides 30% with a 5% defect rate. Supplier C provides 20% with a 1% defect rate. A randomly inspected widget turns out defective. What's the probability it came from Supplier B?

<details>
<summary>Show solution</summary>

Let $A$, $B$, $C$ be "sourced from supplier A/B/C" and $D$ = "defective."

$$P(A)=0.5,\ P(D\mid A)=0.02 \qquad P(B)=0.3,\ P(D\mid B)=0.05 \qquad P(C)=0.2,\ P(D\mid C)=0.01$$

By the law of total probability:

$$P(D) = P(D\mid A)P(A) + P(D\mid B)P(B) + P(D\mid C)P(C) = 0.01 + 0.015 + 0.002 = 0.027$$

By Bayes' theorem:

$$P(B\mid D) = \frac{P(D\mid B)P(B)}{P(D)} = \frac{0.015}{0.027} \approx 0.5556$$

**≈ 55.6%.** Notice Supplier B contributes only 30% of volume but over half of all
defects — its higher per-unit defect rate outweighs its smaller share. This is the same
"weight the rate by the volume" reasoning as the disease-test example in file 01, just
with 3 competing sources instead of a binary sick/healthy split — recognize the pattern
(a partition of the sample space + a law-of-total-probability denominator), not just the
specific numbers.

</details>

### Problem 2 (expected value / linearity of expectation): At a party, $n$ people each throw their hat into a pile, then each grabs one hat back at random (a uniformly random permutation of hats to people). What is the expected number of people who get their own hat back?

<details>
<summary>Show solution</summary>

Define indicator $X_i = 1$ if person $i$ gets their own hat back, else 0. We want
$E\!\left[\sum_{i=1}^n X_i\right]$.

By **linearity of expectation** (which holds regardless of whether the $X_i$ are
independent — and here they are *not* independent, since one person getting their hat
back changes the probabilities for everyone else):

$$E\left[\sum_i X_i\right] = \sum_i E[X_i] = \sum_i P(X_i = 1)$$

Each person has exactly $1/n$ probability of drawing their own hat out of $n$ equally
likely hats, so:

$$E\left[\sum_i X_i\right] = n \cdot \frac{1}{n} = 1$$

**The expected number of matches is exactly 1, regardless of $n$** — whether there are 5
people or 5,000. This is the punchline interviewers are checking for: it would be very
hard to compute directly (it requires the distribution of fixed points of a random
permutation, which involves derangements), but linearity of expectation sidesteps the
dependency structure entirely and makes it a one-line computation.

</details>

### Problem 3 (combinatorics): A project team of 5 people is chosen from a pool of 8 engineers and 4 designers. The team must include at least 2 designers. How many possible teams are there?

<details>
<summary>Show solution</summary>

Two equivalent approaches — do both as a check.

**Complement approach:** total unrestricted teams minus teams with 0 or 1 designer.

$$\binom{12}{5} = 792 \qquad \binom{8}{5}=56 \text{ (0 designers)} \qquad \binom{4}{1}\binom{8}{4} = 4\times70=280 \text{ (1 designer)}$$

$$792 - 56 - 280 = 456$$

**Direct sum approach:** sum over $d = 2, 3, 4$ designers (can't have 5, only 4 designers exist):

$$\binom{4}{2}\binom{8}{3} + \binom{4}{3}\binom{8}{2} + \binom{4}{4}\binom{8}{1} = (6)(56) + (4)(28) + (1)(8) = 336+112+8 = 456$$

Both give **456 teams**. In an interview, computing it both ways (or at least mentioning
you could) is a good way to show you're not just pattern-matching a memorized formula —
it demonstrates you understand *why* the complement trick and the direct-sum both have to
agree.

</details>

### Problem 4 (hypothesis testing on real data): Your A/B test: **Control** had 5,000 users, 410 conversions. **Treatment** had 5,020 users, 460 conversions. Is this a statistically significant lift? What test would you run, and what do you conclude?

<details>
<summary>Show solution</summary>

This is a comparison of two independent proportions — the right test is a **two-proportion
z-test** (equivalently, a chi-square test of independence on the 2×2 table; both give the
same result for a two-sided test).

$$\hat p_1 = 410/5000 = 0.0820 \qquad \hat p_2 = 460/5020 = 0.09163$$

Pooled proportion under $H_0: p_1 = p_2$:

$$\hat p = \frac{410+460}{5000+5020} = \frac{870}{10020} = 0.08683$$

Standard error:

$$SE = \sqrt{\hat p(1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)} = \sqrt{0.08683 \times 0.91317 \times (0.0002+0.0001992)} \approx 0.00563$$

Test statistic:

$$z = \frac{\hat p_2 - \hat p_1}{SE} = \frac{0.00963}{0.00563} \approx 1.71$$

Two-sided p-value $\approx 2(1-\Phi(1.71)) \approx 0.087$.

**Conclusion: not statistically significant at $\alpha=0.05$** ($p \approx 0.087 > 0.05$) —
though it's close enough ("directionally positive, borderline") that the honest next step
is to talk about power: what sample size would you need to reliably detect a lift of this
size (~1.1 percentage points, roughly a 12% relative lift), and is the test simply
underpowered rather than the effect being genuinely null? This is exactly the kind of
follow-up file 02 covers in depth (sample size / MDE calculations) — a good answer here
doesn't stop at "not significant," it says what you'd do next.

</details>

### Problem 5 (confidence interval interpretation): You draw a sample of 100 customers; mean order value is \$85, sample standard deviation is \$20. (a) Compute a 95% CI for the true mean order value. (b) Which of these is the *correct* interpretation: "There's a 95% chance the true mean is between \$81.08 and \$88.92," or "If we repeated this sampling process many times, 95% of the resulting intervals would contain the true mean"? (c) If you quadrupled your sample size to $n=400$ (same mean and std), how would the CI width change?

<details>
<summary>Show solution</summary>

**(a)** $SE = s/\sqrt n = 20/\sqrt{100} = 2$. CI $= 85 \pm 1.96 \times 2 = 85 \pm 3.92 =$
**(\$81.08, \$88.92)**.

**(b)** The **second** statement is the technically correct frequentist interpretation.
The true mean is a fixed (if unknown) number — it either is or isn't in any specific
interval, so it's meaningless to assign it a 95% "probability" after the fact. The 95%
refers to the *procedure*: if you repeated this sampling-and-interval-construction process
many times, 95% of the resulting (different, random) intervals would contain the true
mean. This is the single most commonly misstated fact in applied statistics — see file 02
for the full derivation of why, and the Bayesian credible-interval contrast in file 01.

**(c)** Standard error scales as $1/\sqrt n$, so quadrupling $n$ from 100 to 400 **halves**
the standard error ($2 \to 1$) and therefore **halves the CI width** (from $\pm 3.92$ to
$\pm 1.96$, i.e. total width $7.84 \to 3.92$). This $\sqrt n$ relationship is why "just
collect 10x more data" gives diminishing returns on precision — you need 4x the sample to
halve your margin of error, not 2x.

</details>

### Problem 6 (distribution identification): Match each scenario to the probability distribution that models it, and briefly justify why.

(a) Number of customer support tickets arriving in a fixed 1-hour window, where tickets
arrive independently at a constant average rate.
(b) The waiting time between two consecutive ticket arrivals in that same system.
(c) Number of defective units in a fixed batch of 50, where each unit is independently
defective with probability $p$.
(d) Number of failed login attempts before the first successful one, where each attempt
independently succeeds with probability $p$.
(e) Number of trials needed to observe the 3rd success, in a sequence of independent
trials each with success probability $p$.

<details>
<summary>Show solution</summary>

(a) **Poisson.** Counts of independent events over a fixed interval at a constant average
rate $\lambda$ — the canonical Poisson setup.

(b) **Exponential.** The continuous-time counterpart to a Poisson process: the *gap*
between events in a Poisson process is exponentially distributed. This pairing (Poisson
count ↔ Exponential gap) is one of the most commonly tested "why does this distribution
apply here" facts.

(c) **Binomial.** A *fixed number* of independent identical trials ($n=50$), each with the
same success probability $p$, counting total successes — textbook Binomial.

(d) **Geometric.** Counts trials *until the first* success in a sequence of independent
Bernoulli trials. (Careful with the off-by-one convention: some definitions count the
number of failures before the first success, others count the trial number of the first
success itself — always state which one you're using.)

(e) **Negative Binomial.** The generalization of Geometric to the $r$-th success instead
of the 1st — here $r=3$. Geometric is the special case of Negative Binomial with $r=1$.

</details>

### Problem 7 (brainteaser — expected coin flips to HH): You flip a fair coin repeatedly. What is the expected number of flips until you see two heads in a row (HH)?

<details>
<summary>Show solution</summary>

Set up states by "how far into a streak am I": $S_0$ = no progress (last flip wasn't H,
or we just started), $S_1$ = last flip was H, $S_2$ = done (HH just occurred). Let $E_0,
E_1$ be the expected number of *additional* flips needed from each state.

From $S_0$: flip once (1 flip used). With probability $\tfrac12$ it's H → move to $S_1$;
with probability $\tfrac12$ it's T → stay at $S_0$.

$$E_0 = 1 + \tfrac12 E_1 + \tfrac12 E_0 \implies \tfrac12 E_0 = 1 + \tfrac12 E_1 \implies E_0 = 2 + E_1$$

From $S_1$: flip once. With probability $\tfrac12$ it's H → done (0 more flips needed);
with probability $\tfrac12$ it's T → back to $S_0$.

$$E_1 = 1 + \tfrac12(0) + \tfrac12 E_0 = 1 + \tfrac12 E_0$$

Substitute:

$$E_0 = 2 + \left(1 + \tfrac12 E_0\right) = 3 + \tfrac12 E_0 \implies \tfrac12 E_0 = 3 \implies E_0 = 6$$

**Expected number of flips to see HH is 6.** (Fun follow-up if asked: the expected number
of flips to see HT, by the same method, is only **4** — same fair coin, but HT is "easier"
to wait for than HH, because once you see an H waiting for T, a T failure doesn't erase
your progress the way it does for HH, where a single T sends you all the way back to
square one. This asymmetry surprises most people and is a good thing to mention if the
interviewer asks a follow-up.)

</details>

### Problem 8 (St. Petersburg paradox): A casino offers this game: flip a fair coin repeatedly. The pot starts at \$2 and doubles every time you flip heads. The moment you flip tails, the game ends and you win whatever is in the pot. What is the expected payout? Is that a sensible price to charge for entry, and if not, why not?

<details>
<summary>Show solution</summary>

If the first tails occurs on flip $k+1$ (i.e., you saw $k$ heads first, each with
probability $(1/2)^k$, followed by one tails), your payout is $2^k$ dollars... more
precisely under the "doubles per head" framing, payout after $k$ heads is $2^k$ and that
specific outcome (k heads then a tail) has probability $(1/2)^{k+1}$. Summing over all
possible $k \geq 0$:

$$E[\text{payout}] = \sum_{k=0}^{\infty} \left(\frac12\right)^{k+1} \cdot 2^{k} = \sum_{k=0}^{\infty} \frac12 = \infty$$

**The expected payout is literally infinite** — each term in the sum contributes exactly
$\tfrac12$, and there are infinitely many terms. Yet virtually no rational person would
pay more than perhaps \$10–20 to play this game once.

**This is the paradox, and the resolution is expected *utility*, not expected *value*.**
Raw dollar expected value treats the millionth dollar as worth exactly as much as the
first, which isn't how anyone actually values money — marginal utility of money
diminishes. If you instead maximize expected utility with a concave utility function (e.g.
Bernoulli's proposed $u(w) = \log(w)$), the expected utility of this game is finite, and
the "certainty equivalent" (the guaranteed cash amount with the same utility as playing)
comes out to a small, sensible number. This is the historical origin of expected-utility
theory in economics — it exists specifically because raw expected value gives absurd
answers for high-variance, unbounded-payout bets like this one, and it's a good thing to
bring up if you're ever asked about risk-adjusted decision-making or Kelly criterion-style
questions.

</details>

### Problem 9 (Bertrand's box paradox): There are 3 boxes. Box 1 has 2 gold coins. Box 2 has 1 gold and 1 silver coin. Box 3 has 2 silver coins. You pick a box uniformly at random, then draw one coin from it uniformly at random. It's gold. What's the probability the *other* coin in that same box is also gold?

<details>
<summary>Show solution</summary>

The naive (wrong) answer is $1/2$ — "it's either Box 1 or Box 2, so 50/50." This ignores
that Box 1 is *twice as likely* to have produced a gold draw as Box 2, since both of its
coins are gold.

Use Bayes' theorem properly. Let $G$ = "drew a gold coin."

$$P(G\mid\text{Box 1}) = 1 \qquad P(G\mid\text{Box 2}) = \tfrac12 \qquad P(G\mid\text{Box 3}) = 0$$

$$P(G) = 1\cdot\tfrac13 + \tfrac12\cdot\tfrac13 + 0\cdot\tfrac13 = \tfrac13 + \tfrac16 = \tfrac12$$

$$P(\text{Box 1}\mid G) = \frac{P(G\mid\text{Box 1})P(\text{Box 1})}{P(G)} = \frac{1\cdot\tfrac13}{\tfrac12} = \frac23$$

Since Box 1 is the only box where the *other* coin is guaranteed gold, and Box 3 is
impossible given we drew gold, $P(\text{other coin is gold}) = P(\text{Box 1}\mid G) =
\mathbf{2/3}$.

**Intuition for why it's not 1/2:** think of the boxes as containing 6 individually
labeled coins instead of 3 boxes: Box1-CoinA(gold), Box1-CoinB(gold), Box2-CoinA(gold),
Box2-CoinB(silver), Box3-CoinA(silver), Box3-CoinB(silver). All 6 coins are equally likely
to be "the one you drew." Exactly 3 are gold: Box1-A, Box1-B, Box2-A — and *two out of
those three* gold coins live in Box 1 (where the partner is also gold). $2/3$, not $1/2$.
This relabeling trick — expanding "pick a box then a coin" into "pick one of 6 equally
likely coins directly" — is the cleanest way to make the counterintuitive answer feel
obvious, and it generalizes to Monty-Hall-style problems too.

</details>

---

## Part 3 — "Derive It From Scratch" Prompts

Do these on paper before expanding. They're meant to be quicker refreshers than the full
walkthroughs in the companion files — if you get stuck, the file/section pointer tells you
exactly where to go for the complete derivation with more context and intuition.

### Prompt 1: Derive the closed-form OLS solution from the sum-of-squared-residuals objective.

<details>
<summary>Show solution</summary>

Objective: minimize $J(\beta) = (y - X\beta)^\top(y-X\beta)$ over $\beta$.

Expand:

$$J(\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X \beta$$

Take the gradient with respect to $\beta$ and set to zero:

$$\nabla_\beta J = -2X^\top y + 2X^\top X\beta = 0$$

$$X^\top X \beta = X^\top y \implies \boxed{\beta = (X^\top X)^{-1}X^\top y}$$

(assuming $X^\top X$ is invertible, i.e. no perfect multicollinearity). Second-order check:
the Hessian is $2X^\top X$, which is positive semi-definite for any $X$, so this is a
global minimum (a convex objective), not just a stationary point. Full derivation with
geometric intuition (projection onto the column space of $X$) in file 03, Section 3.

</details>

### Prompt 2: Derive the sigmoid function's gradient, then the gradient of log-loss with respect to the logistic regression weights.

<details>
<summary>Show solution</summary>

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Using the quotient rule (or rewriting as $\sigma(z) = (1+e^{-z})^{-1}$ and the chain rule):

$$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)\big(1-\sigma(z)\big)$$

Now log-loss for a single example, $\hat y = \sigma(z)$, $z = w^\top x + b$:

$$L = -\big[y\log\hat y + (1-y)\log(1-\hat y)\big]$$

Chain rule, $\dfrac{\partial L}{\partial z} = \dfrac{\partial L}{\partial \hat y}\cdot\dfrac{\partial \hat y}{\partial z}$:

$$\frac{\partial L}{\partial \hat y} = -\frac{y}{\hat y} + \frac{1-y}{1-\hat y} = \frac{\hat y - y}{\hat y(1-\hat y)}$$

$$\frac{\partial \hat y}{\partial z} = \hat y(1-\hat y)$$

Multiply — the $\hat y(1-\hat y)$ terms cancel exactly:

$$\frac{\partial L}{\partial z} = \frac{\hat y - y}{\hat y(1-\hat y)} \cdot \hat y(1-\hat y) = \hat y - y$$

This clean cancellation is the whole reason logistic regression pairs log-loss with the
sigmoid specifically — the gradient reduces to the simple, interpretable "prediction minus
truth" residual. Then by the chain rule through $z = w^\top x + b$:

$$\frac{\partial L}{\partial w} = (\hat y - y)\,x \qquad \frac{\partial L}{\partial b} = \hat y - y$$

Full derivation and the connection to the cross-entropy/maximum-likelihood view in file 03,
Section 6.

</details>

### Prompt 3: Derive Bayes' theorem from the definition of conditional probability.

<details>
<summary>Show solution</summary>

Start from the definition of conditional probability (this *is* the definition, not a
derived fact):

$$P(A\mid B) = \frac{P(A\cap B)}{P(B)} \qquad P(B\mid A) = \frac{P(A\cap B)}{P(A)}, \quad P(A),P(B)>0$$

Both equations share the same numerator, $P(A\cap B)$. Solve each for it:

$$P(A\cap B) = P(A\mid B)\,P(B) = P(B\mid A)\,P(A)$$

Take the right-hand equality and divide both sides by $P(B)$:

$$\boxed{P(A\mid B) = \frac{P(B\mid A)\,P(A)}{P(B)}}$$

If $P(B)$ isn't given directly, expand it via the law of total probability over a
partition $\{A_i\}$ of the sample space: $P(B) = \sum_i P(B\mid A_i)P(A_i)$ — this is the
denominator you compute in every applied Bayes word problem (see Part 2, Problem 1 above).
Full treatment in file 01, Section 5.

</details>

### Prompt 4: Derive the bias-variance decomposition of expected test MSE.

<details>
<summary>Show solution</summary>

Assume the true relationship is $y = f(x) + \varepsilon$, with $E[\varepsilon]=0$,
$\mathrm{Var}(\varepsilon)=\sigma^2$, and $\varepsilon$ independent of $x$. Let $\hat f(x)$
be a model fit on random training data (so $\hat f(x)$ itself is a random variable across
different training sets). Expected squared error at a fixed test point $x$:

$$E\big[(y-\hat f(x))^2\big] = E\big[(f(x)+\varepsilon - \hat f(x))^2\big]$$

Add and subtract $E[\hat f(x)]$ inside the square:

$$= E\Big[\big(\varepsilon + (f(x)-E[\hat f(x)]) + (E[\hat f(x)] - \hat f(x))\big)^2\Big]$$

Expand the square into three squared terms plus three cross terms. Every cross term
vanishes: $\varepsilon$ is independent of $x$ and zero-mean, and $(f(x)-E[\hat f(x)])$ is a
constant (no randomness left once we've taken the expectation inside $E[\hat f(x)]$) while
$(E[\hat f(x)]-\hat f(x))$ has mean zero by construction. What survives:

$$= \underbrace{E[\varepsilon^2]}_{\sigma^2}
+ \underbrace{\big(f(x)-E[\hat f(x)]\big)^2}_{\text{Bias}^2}
+ \underbrace{E\big[(\hat f(x)-E[\hat f(x)])^2\big]}_{\text{Variance}}$$

$$\boxed{\text{Expected Test MSE} = \sigma^2_{\text{irreducible}} + \text{Bias}(\hat f(x))^2 + \text{Var}(\hat f(x))}$$

Full derivation with the intuition for why increasing model complexity trades bias for
variance (and the classic U-shaped total-error curve) in file 03, Section 1.

</details>

### Prompt 5: Derive why Ridge regression's closed-form solution is $(X^\top X + \lambda I)^{-1}X^\top y$.

<details>
<summary>Show solution</summary>

Ridge objective adds an L2 penalty to the OLS objective:

$$J(\beta) = (y-X\beta)^\top(y-X\beta) + \lambda\,\beta^\top\beta$$

Gradient with respect to $\beta$ (reusing the OLS expansion from Prompt 1, plus the
derivative of $\lambda\beta^\top\beta$, which is $2\lambda\beta$):

$$\nabla_\beta J = -2X^\top y + 2X^\top X\beta + 2\lambda\beta = 0$$

$$X^\top X\beta + \lambda\beta = X^\top y \implies (X^\top X + \lambda I)\beta = X^\top y$$

$$\boxed{\beta = (X^\top X + \lambda I)^{-1}X^\top y}$$

Two things worth being able to say out loud about this result: (1) adding $\lambda I$
(a positive value on every diagonal entry) makes the matrix invertible even when $X^\top X$
is singular or near-singular from multicollinearity — this is the "ridge" in Ridge
regression, literally a ridge added to the diagonal; (2) as $\lambda \to \infty$, $\beta \to
0$ (maximal shrinkage), and as $\lambda \to 0$, this reduces exactly to the OLS solution
from Prompt 1. Full derivation plus the Bayesian interpretation (Ridge = OLS with a
Gaussian prior on $\beta$) in file 03, Section 2.

</details>

### Prompt 6: Derive the XGBoost optimal leaf weight from the regularized objective.

<details>
<summary>Show solution</summary>

At boosting round $t$, using the 2nd-order Taylor expansion of the loss around the
current prediction (with $g_i, h_i$ the first/second-order gradients of the loss for
example $i$), the regularized objective for a candidate tree $f_t$ with $T$ leaves and
leaf weights $w_j$ is:

$$\text{Obj} = \sum_{i=1}^n \Big[g_i f_t(x_i) + \tfrac12 h_i f_t(x_i)^2\Big] + \gamma T + \tfrac12\lambda\sum_{j=1}^T w_j^2$$

Since every sample $i$ falls into exactly one leaf, regroup the sum by leaf. Let $I_j$ =
set of sample indices assigned to leaf $j$, $G_j = \sum_{i\in I_j} g_i$, $H_j = \sum_{i\in
I_j} h_i$:

$$\text{Obj} = \sum_{j=1}^T \Big[ G_j w_j + \tfrac12(H_j+\lambda) w_j^2 \Big] + \gamma T$$

This is now $T$ **independent** 1-D quadratics in $w_j$ (one per leaf) — a quadratic of
the form $Gw + \tfrac12(H+\lambda)w^2$, minimized where its derivative is zero:

$$\frac{\partial}{\partial w_j}\Big[G_j w_j + \tfrac12(H_j+\lambda)w_j^2\Big] = G_j + (H_j+\lambda)w_j = 0$$

$$\boxed{w_j^* = -\frac{G_j}{H_j + \lambda}}$$

Plugging $w_j^*$ back into the objective gives the optimal objective value per leaf,
$-\tfrac12\frac{G_j^2}{H_j+\lambda}$, which summed across leaves (plus $\gamma T$) is
exactly the quantity used to score candidate splits (the "gain" formula) — see file 04,
Section on XGBoost internals, for the full split-gain derivation building on this result.

</details>

### Prompt 7: Derive the gradient of softmax cross-entropy loss with respect to the pre-softmax logits.

<details>
<summary>Show solution</summary>

Softmax turns logits $z_1,\dots,z_K$ into probabilities:

$$p_k = \frac{e^{z_k}}{\sum_{m} e^{z_m}}$$

Cross-entropy loss against a one-hot true label $y$ (true class $c$, so $y_c=1$ and
$y_k=0$ for $k\neq c$):

$$L = -\sum_k y_k \log p_k = -\log p_c$$

First, the Jacobian of softmax itself (a standard but easy-to-forget result): differentiate
$p_k$ with respect to $z_j$ using the quotient rule, splitting into the $k=j$ and $k\neq j$
cases, both of which collapse into one formula using the Kronecker delta $\delta_{kj}$:

$$\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$$

Now apply the chain rule for $\partial L/\partial z_j$:

$$\frac{\partial L}{\partial z_j} = \sum_k \frac{\partial L}{\partial p_k}\cdot\frac{\partial p_k}{\partial z_j} = \sum_k \left(-\frac{y_k}{p_k}\right)\cdot p_k(\delta_{kj}-p_j)$$

The $p_k$ terms cancel inside the sum:

$$= \sum_k -y_k(\delta_{kj} - p_j) = -\sum_k y_k\delta_{kj} + p_j\sum_k y_k = -y_j + p_j\cdot 1$$

(using $\sum_k y_k = 1$ since $y$ is one-hot):

$$\boxed{\frac{\partial L}{\partial z_j} = p_j - y_j}$$

Same elegant "prediction minus truth" form as the binary sigmoid+log-loss case in Prompt
2 — this is not a coincidence, softmax cross-entropy is the multi-class generalization of
exactly that setup, and it's why both are the default loss/activation pairing for
classification: the combination always produces this clean gradient regardless of how
many classes there are. See file 11 for where this gradient plugs into backprop through a
neural net's final layer.

</details>

### Prompt 8: Derive the variance of a sum of (possibly correlated) random variables.

<details>
<summary>Show solution</summary>

Start from the definition of variance applied to $X+Y$:

$$\mathrm{Var}(X+Y) = E\big[(X+Y - E[X+Y])^2\big] = E\big[\big((X-E[X]) + (Y-E[Y])\big)^2\big]$$

Expand the square:

$$= E\big[(X-E[X])^2\big] + E\big[(Y-E[Y])^2\big] + 2E\big[(X-E[X])(Y-E[Y])\big]$$

The first two terms are $\mathrm{Var}(X)$ and $\mathrm{Var}(Y)$ by definition; the third is
$2\,\mathrm{Cov}(X,Y)$ by definition:

$$\boxed{\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y)}$$

Generalizing to $n$ variables (same expansion, more cross terms):

$$\mathrm{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \mathrm{Var}(X_i) + \sum_{i\neq j}\mathrm{Cov}(X_i,X_j)$$

For a weighted sum $\sum_i w_i X_i$ (e.g. a portfolio of assets, or a weighted ensemble of
model predictions), this generalizes cleanly to matrix form: if $\Sigma$ is the
covariance matrix of the $X_i$ and $w$ is the weight vector, $\mathrm{Var}(w^\top X) =
w^\top \Sigma w$. Two practical consequences worth stating out loud: if all pairwise
correlations are zero, variance of the sum is just the sum of variances (this is *why*
averaging $n$ i.i.d. predictions divides variance by $n$ — the classic bagging/ensemble
argument in file 04); if variables are positively correlated, the sum's variance is
strictly larger than the sum of individual variances, which is exactly why diversifying
into *uncorrelated* (not just "more") assets/models actually reduces risk/variance.

</details>

---

## Part 4 — Runnable Code Exercises

Each script below is fully self-contained — fixed random seed, synthetic data generated
inline, no external files. Copy-paste and run as-is; each should finish in well under a
few seconds.

### Exercise 1: Implement k-means from scratch in NumPy, and verify it against scikit-learn's `KMeans` on synthetic blob data.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(42)

# Generate 4 well-separated 2D clusters
X, true_labels = make_blobs(
    n_samples=300, centers=4, n_features=2, cluster_std=1.0, random_state=42
)


def kmeans_from_scratch(X, k, n_iters=100, random_state=42):
    """Lloyd's algorithm: alternate assignment and update steps until convergence."""
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]

    # Initialize centroids by picking k random data points (no fancy k-means++ here)
    init_idx = rng.choice(n_samples, size=k, replace=False)
    centroids = X[init_idx].copy()

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(n_iters):
        # Assignment step: distance from every point to every centroid, shape (n, k)
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Update step: each centroid becomes the mean of its assigned points
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])

        if np.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids, labels


def align_centroids(centroids_a, centroids_b):
    """K-means cluster IDs are arbitrary across runs/implementations - greedily match
    each centroid in `centroids_a` to its nearest counterpart in `centroids_b` so we can
    compare them position-by-position."""
    order, used = [], set()
    for c in centroids_b:
        dists = np.linalg.norm(centroids_a - c, axis=1)
        for idx in np.argsort(dists):
            if idx not in used:
                order.append(idx)
                used.add(idx)
                break
    return centroids_a[order]


k = 4
scratch_centroids, scratch_labels = kmeans_from_scratch(X, k)

sklearn_km = KMeans(n_clusters=k, n_init=10, random_state=42)
sklearn_labels = sklearn_km.fit_predict(X)
sklearn_centroids = sklearn_km.cluster_centers_

aligned_scratch_centroids = align_centroids(scratch_centroids, sklearn_centroids)

print("sklearn centroids:\n", np.round(sklearn_centroids, 3))
print("\nScratch centroids (aligned to sklearn's ordering):\n",
      np.round(aligned_scratch_centroids, 3))
print("\nMax centroid distance (scratch vs sklearn):",
      round(np.max(np.linalg.norm(aligned_scratch_centroids - sklearn_centroids, axis=1)), 4))
# Expect this distance to be very small (near 0) on well-separated blobs like these -
# if k-means converges to a different local optimum you may see a larger gap, which is
# expected behavior of the algorithm (sensitivity to initialization), not a bug.
```

### Exercise 2: Implement batch gradient descent for linear regression from scratch and compare the learned coefficients to scikit-learn's closed-form solution.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)

n_samples, n_features = 200, 3
X = np.random.randn(n_samples, n_features)
true_coef = np.array([3.5, -2.0, 1.2])
true_intercept = 4.0
noise = np.random.normal(0, 0.5, n_samples)
y = X @ true_coef + true_intercept + noise


def gradient_descent_linear_regression(X, y, lr=0.1, n_iters=2000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(n_iters):
        y_pred = X @ w + b
        error = y_pred - y                       # shape (n_samples,)
        # Gradient of MSE = mean(error^2) w.r.t. w and b
        grad_w = (2.0 / n_samples) * (X.T @ error)
        grad_b = (2.0 / n_samples) * np.sum(error)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


w_gd, b_gd = gradient_descent_linear_regression(X, y, lr=0.1, n_iters=2000)

sklearn_model = LinearRegression()
sklearn_model.fit(X, y)

print("True coefficients:      ", true_coef, " intercept:", true_intercept)
print("Gradient descent coefs: ", np.round(w_gd, 4), " intercept:", round(b_gd, 4))
print("sklearn (closed-form):  ", np.round(sklearn_model.coef_, 4),
      " intercept:", round(sklearn_model.intercept_, 4))
print("\nMax abs difference between GD and sklearn coefficients:",
      round(float(np.max(np.abs(w_gd - sklearn_model.coef_))), 6))
# With standardized-scale synthetic features like these, batch GD at lr=0.1 for 2000
# iterations converges to within floating-point noise of the closed-form OLS solution.
```

### Exercise 3: Fit an XGBoost classifier on synthetic data, compute SHAP values, and print the top features by mean |SHAP value|.

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend - saves to file instead of popping a window
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

np.random.seed(42)

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42
)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    eval_metric="logloss", random_state=42
)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

# TreeExplainer is exact and fast for tree ensembles (no sampling/approximation needed)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)          # shap.Explanation object, .values shape (n, n_features)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_order = np.argsort(mean_abs_shap)[::-1]

print("\nTop features by mean |SHAP value|:")
for rank, idx in enumerate(importance_order[:5], start=1):
    print(f"{rank}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

# Bar chart of global feature importance, saved to a file
plt.figure(figsize=(8, 5))
sorted_importances = mean_abs_shap[importance_order]
sorted_names = [feature_names[i] for i in importance_order]
plt.barh(sorted_names[::-1], sorted_importances[::-1])
plt.xlabel("Mean |SHAP value|")
plt.title("Global feature importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=100)
print("\nSaved plot to shap_feature_importance.png")
```

### Exercise 4: Implement logistic regression's gradient descent from scratch (sigmoid + log-loss + gradient update) and compare to scikit-learn's `LogisticRegression`.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

np.random.seed(42)

X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3, n_redundant=0, random_state=42
)


def sigmoid(z):
    z = np.clip(z, -500, 500)   # avoid overflow warnings for very large |z|
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.5, n_iters=3000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(n_iters):
        z = X @ w + b
        y_hat = sigmoid(z)
        error = y_hat - y                        # dL/dz = y_hat - y (derived in Part 3, Prompt 2)
        grad_w = (X.T @ error) / n_samples
        grad_b = np.mean(error)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


w_scratch, b_scratch = train_logistic_regression(X, y)

sklearn_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
sklearn_model.fit(X, y)

print("Scratch weights:  ", np.round(w_scratch, 4))
print("sklearn weights:  ", np.round(sklearn_model.coef_[0], 4))
print("Scratch intercept:", round(b_scratch, 4))
print("sklearn intercept:", round(sklearn_model.intercept_[0], 4))

scratch_preds = (sigmoid(X @ w_scratch + b_scratch) >= 0.5).astype(int)
sklearn_preds = sklearn_model.predict(X)
agreement = np.mean(scratch_preds == sklearn_preds)
print(f"\nPrediction agreement between scratch and sklearn: {agreement * 100:.1f}%")
# penalty=None disables sklearn's default L2 regularization so both models are optimizing
# the exact same unregularized objective, making the coefficients directly comparable.
```

### Exercise 5: Walk-forward time-series cross-validation using `TimeSeriesSplit` on synthetic seasonal data, fitting a model per fold and reporting per-fold MAE.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# Synthetic daily series: trend + monthly-ish and weekly seasonality + noise
n = 500
t = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * t / 30) + 5 * np.sin(2 * np.pi * t / 7)
trend = 0.03 * t
noise = np.random.normal(0, 2, n)
y = 100 + trend + seasonal + noise

df = pd.DataFrame({"y": y})
df["lag_1"] = df["y"].shift(1)
df["lag_7"] = df["y"].shift(7)
df["lag_30"] = df["y"].shift(30)
df["roll_mean_7"] = df["y"].shift(1).rolling(7).mean()
df = df.dropna().reset_index(drop=True)   # drop warm-up rows with missing lag/rolling values

feature_cols = ["lag_1", "lag_7", "lag_30", "roll_mean_7"]
X = df[feature_cols].values
y_target = df["y"].values

tscv = TimeSeriesSplit(n_splits=5)
fold_maes = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_target[train_idx], y_target[test_idx]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    fold_maes.append(mae)
    print(f"Fold {fold}: train_size={len(train_idx):4d}, test_size={len(test_idx):3d}, MAE={mae:.3f}")

print(f"\nAverage MAE across folds: {np.mean(fold_maes):.3f} (+/- {np.std(fold_maes):.3f})")
# TimeSeriesSplit never lets a fold's test set precede its train set in time - this is the
# walk-forward property that plain k-fold CV violates for temporal data (see file 07).
```

### Exercise 6: A pandas exercise — compute lag features, rolling means, and Fourier-term seasonality features for a synthetic daily time series.

```python
import numpy as np
import pandas as pd

np.random.seed(42)

n = 400
dates = pd.date_range("2023-01-01", periods=n, freq="D")
t = np.arange(n)

# Trend + weekly seasonality + annual seasonality + noise
trend = 0.05 * t
weekly_season = 5 * np.sin(2 * np.pi * t / 7)
annual_season = 10 * np.sin(2 * np.pi * t / 365.25)
noise = np.random.normal(0, 2, n)
value = 50 + trend + weekly_season + annual_season + noise

df = pd.DataFrame({"date": dates, "value": value}).set_index("date")

# Lag features
df["lag_1"] = df["value"].shift(1)
df["lag_7"] = df["value"].shift(7)

# Rolling-mean features - shift(1) first so the window never includes the current row
# (using the current row's own value as a "feature" would be leakage at prediction time)
df["roll_mean_7"] = df["value"].shift(1).rolling(window=7).mean()
df["roll_mean_28"] = df["value"].shift(1).rolling(window=28).mean()

# Fourier terms: K=2 harmonics each for weekly (period=7) and annual (period=365.25)
# seasonality - a compact alternative to one-hot day-of-week/month dummies that lets a
# linear model capture smooth cyclical patterns with few extra columns
t_idx = np.arange(len(df))
for k in range(1, 3):
    df[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * t_idx / 7)
    df[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * t_idx / 7)
    df[f"fourier_annual_sin_{k}"] = np.sin(2 * np.pi * k * t_idx / 365.25)
    df[f"fourier_annual_cos_{k}"] = np.cos(2 * np.pi * k * t_idx / 365.25)

df = df.dropna()  # drop the warm-up rows where lag_7/roll_mean_28 aren't yet defined
print(df.head(10))
print("\nShape after dropping warm-up rows:", df.shape)
```

---

## Part 5 — ML/Coding Whiteboard Questions

### Question 1: Write a function that computes precision, recall, and F1 from `y_true`/`y_pred` arrays of 0/1 labels — no scikit-learn.

<details>
<summary>Show solution</summary>

```python
def precision_recall_f1(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
print(precision_recall_f1(y_true, y_pred))  # -> (0.75, 0.75, 0.75)
```

The guard clauses (returning 0.0 when the denominator is 0) matter — e.g. a model that
never predicts positive gives `tp+fp=0`, which would otherwise be a division-by-zero
crash rather than the conventionally-defined "precision is undefined/0 here."

</details>

### Question 2: Write a function that computes the Gini impurity of a split, given the class counts on each side.

<details>
<summary>Show solution</summary>

```python
def gini_impurity(class_counts):
    """class_counts: e.g. [n_class0, n_class1, ...]"""
    total = sum(class_counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in class_counts]
    return 1 - sum(p ** 2 for p in probs)


def weighted_gini_of_split(left_counts, right_counts):
    n_left, n_right = sum(left_counts), sum(right_counts)
    n_total = n_left + n_right
    return (
        (n_left / n_total) * gini_impurity(left_counts)
        + (n_right / n_total) * gini_impurity(right_counts)
    )


# 10 samples (6 class-0, 4 class-1) split into left=[5,1], right=[1,3]
print(weighted_gini_of_split([5, 1], [1, 3]))  # -> ~0.3167
```

A tree-building algorithm evaluates this weighted Gini for every candidate split and
picks the one that minimizes it (equivalently, maximizes the *Gini gain* versus the
parent node's impurity) — see file 04 for the full split-gain formula this generalizes
into for boosted trees.

</details>

### Question 3: Write a function implementing an exponentially-weighted moving average (EWMA).

<details>
<summary>Show solution</summary>

```python
def ewma(values, alpha):
    """Exponentially weighted moving average.
    alpha in (0, 1]: higher alpha weights recent observations more heavily
    (alpha=1 reduces to just returning the raw series).
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    result = []
    prev = None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        result.append(prev)
    return result


print(ewma([10, 12, 13, 12, 15, 20], alpha=0.3))
# -> [10, 10.6, 11.32, 11.524, 12.5668, 14.79676]
```

Note the recursive structure: $S_t = \alpha x_t + (1-\alpha)S_{t-1}$, seeded with $S_1 =
x_1$. This is the same recurrence behind EWMA-based volatility estimates, Adam's moment
estimates in deep learning optimizers, and simple exponential smoothing forecasts (file
07) — same formula, different name depending on the field.

</details>

### Question 4: Write a function that detects the "gaps and islands" pattern in a list of dates, in pure Python (no SQL).

<details>
<summary>Show solution</summary>

```python
from datetime import date, timedelta


def find_islands(dates):
    """Given a list of date objects, return (start, end) tuples for each maximal run
    of consecutive calendar days present in `dates`."""
    if not dates:
        return []

    sorted_dates = sorted(set(dates))
    islands = []
    start = prev = sorted_dates[0]

    for d in sorted_dates[1:]:
        if d - prev == timedelta(days=1):
            prev = d                       # extend the current island
        else:
            islands.append((start, prev))  # close current island, start a new one
            start = prev = d

    islands.append((start, prev))          # close the final island
    return islands


def find_gaps(dates):
    """Return (gap_start, gap_end) date ranges that fall strictly between islands."""
    islands = find_islands(dates)
    gaps = []
    for (_, end_prev), (start_next, _) in zip(islands, islands[1:]):
        gaps.append((end_prev + timedelta(days=1), start_next - timedelta(days=1)))
    return gaps


order_dates = [
    date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),
    date(2024, 1, 10), date(2024, 1, 11),
    date(2024, 1, 20),
]

print("Islands:", find_islands(order_dates))
# -> [(2024-01-01, 2024-01-03), (2024-01-10, 2024-01-11), (2024-01-20, 2024-01-20)]
print("Gaps:", find_gaps(order_dates))
# -> [(2024-01-04, 2024-01-09), (2024-01-12, 2024-01-19)]
```

This is the exact same `month_idx - rn` idea from Part 1's SQL gaps-and-islands problems
(11 and 12), just walked linearly instead of using a window-function trick — sorting once
and scanning for breaks in consecutiveness is the pure-Python equivalent of the SQL
`ROW_NUMBER()` subtraction trick.

</details>

### Question 5: Write a function that finds the best split threshold for a single continuous feature, using Gini impurity, via brute-force search over candidate thresholds.

<details>
<summary>Show solution</summary>

```python
def best_split_gini(feature_values, labels):
    def gini(counts):
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts]
        return 1 - sum(p ** 2 for p in probs)

    pairs = sorted(zip(feature_values, labels))
    values_sorted = [p[0] for p in pairs]
    labels_sorted = [p[1] for p in pairs]
    classes = set(labels)
    n = len(values_sorted)

    best_threshold, best_gini = None, float("inf")

    # Candidate thresholds are midpoints between consecutive DISTINCT sorted values -
    # there's never a reason to consider a threshold between two equal values
    for i in range(1, n):
        if values_sorted[i] == values_sorted[i - 1]:
            continue
        threshold = (values_sorted[i] + values_sorted[i - 1]) / 2

        left_labels, right_labels = labels_sorted[:i], labels_sorted[i:]
        left_counts = [left_labels.count(c) for c in classes]
        right_counts = [right_labels.count(c) for c in classes]

        weighted = (
            (len(left_labels) / n) * gini(left_counts)
            + (len(right_labels) / n) * gini(right_counts)
        )
        if weighted < best_gini:
            best_gini, best_threshold = weighted, threshold

    return best_threshold, best_gini


X = [1, 2, 3, 6, 7, 8, 4, 5, 9, 10]
y = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
threshold, gini_score = best_split_gini(X, y)
print(f"Best threshold: {threshold}, weighted Gini: {gini_score:.4f}")
# -> Best threshold: 5.5, weighted Gini: 0.0000
# (the data is perfectly separable at x=5.5: values 1-5 are all class 0, 6-10 all class 1)
```

This brute-force scan (sort once, then walk the sorted values considering only midpoints
between distinct values as candidate thresholds) is exactly the core loop inside a real
decision tree's split-finding routine for a continuous feature — a full tree just repeats
this per feature at every node, keeps the best (feature, threshold) pair, and recurses.

</details>

---

That's the full practice set — 12 SQL problems, 9 probability/stats problems, 8 derivations,
6 runnable code exercises, and 5 whiteboard-coding questions. If you found yourself
reaching for a companion file mid-problem more than once or twice, that's a normal and
useful signal, not a failure — go reread that section, then come back and retry the
problem cold before moving on.
