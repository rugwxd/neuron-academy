"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function SQLChallenges() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          SQL is the most universally tested skill in data science interviews. Every company — from FAANG
          to startups — has a SQL round. The questions test your ability to think in sets, handle edge cases,
          and write efficient queries under time pressure.
        </p>
        <p>
          The 30+ challenges below are organized by difficulty and pattern. They cover the most common
          interview themes: aggregation with GROUP BY, window functions (ROW_NUMBER, LAG, running totals),
          self-joins, date manipulation, handling NULLs, and complex subqueries. Each problem has a
          realistic business context, the expected query, and an explanation of why it works.
        </p>
        <p>
          The schemas are kept simple so you can focus on the SQL logic. In a real interview, always
          clarify the schema, ask about NULL handling, and discuss whether the interviewer wants
          exact duplicates handled.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Key SQL Concepts Refresher</h3>
        <p><strong>Order of operations in SQL:</strong></p>
        <BlockMath math="\text{FROM} \to \text{WHERE} \to \text{GROUP BY} \to \text{HAVING} \to \text{SELECT} \to \text{ORDER BY} \to \text{LIMIT}" />
        <p>
          Understanding this order explains many puzzles: why you cannot use a SELECT alias in WHERE
          (WHERE executes first), why HAVING can filter aggregates (it runs after GROUP BY), and why
          window functions can appear in SELECT but not WHERE.
        </p>

        <h3>Window Function Anatomy</h3>
        <BlockMath math="\text{function}(\text{expr}) \;\text{OVER}\; (\text{PARTITION BY } p \;\text{ORDER BY } o \;\text{ROWS/RANGE frame})" />
        <p>
          The window frame defines which rows contribute to the computation for each row. The default
          frame is <code>RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW</code> when ORDER BY is specified.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Schema for All Problems</h3>
        <CodeBlock
          language="sql"
          title="schema.sql"
          code={`-- Users table
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    name VARCHAR(100),
    signup_date DATE,
    country VARCHAR(50)
);

-- Orders table
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    order_date DATE,
    amount DECIMAL(10,2),
    status VARCHAR(20)  -- 'completed', 'cancelled', 'pending'
);

-- Products table
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2)
);

-- Order items (many-to-many)
CREATE TABLE order_items (
    order_id INT REFERENCES orders(order_id),
    product_id INT REFERENCES products(product_id),
    quantity INT,
    unit_price DECIMAL(10,2)
);

-- Employee hierarchy
CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(100),
    manager_id INT REFERENCES employees(emp_id),
    department VARCHAR(50),
    salary DECIMAL(10,2),
    hire_date DATE
);

-- User logins
CREATE TABLE logins (
    user_id INT REFERENCES users(user_id),
    login_date DATE,
    session_duration_min INT
);`}
        />

        <h3>Easy: Aggregation and Filtering (Problems 1-8)</h3>

        <CodeBlock
          language="sql"
          title="problem_01_top_spenders.sql"
          code={`-- Problem 1: Top 5 users by total spend (completed orders only)
SELECT u.user_id,
       u.name,
       SUM(o.amount) AS total_spent
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE o.status = 'completed'
GROUP BY u.user_id, u.name
ORDER BY total_spent DESC
LIMIT 5;`}
        />

        <CodeBlock
          language="sql"
          title="problem_02_monthly_revenue.sql"
          code={`-- Problem 2: Monthly revenue with month-over-month growth %
WITH monthly AS (
    SELECT DATE_TRUNC('month', order_date) AS month,
           SUM(amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY 1
)
SELECT month,
       revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_month,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 2) AS growth_pct
FROM monthly
ORDER BY month;`}
        />

        <CodeBlock
          language="sql"
          title="problem_03_never_ordered.sql"
          code={`-- Problem 3: Users who signed up but never placed an order
SELECT u.user_id, u.name, u.signup_date
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE o.order_id IS NULL;

-- Alternative with NOT EXISTS (often faster):
SELECT u.user_id, u.name, u.signup_date
FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.user_id
);`}
        />

        <CodeBlock
          language="sql"
          title="problem_04_duplicate_orders.sql"
          code={`-- Problem 4: Find duplicate orders (same user, same date, same amount)
SELECT user_id, order_date, amount, COUNT(*) AS num_duplicates
FROM orders
GROUP BY user_id, order_date, amount
HAVING COUNT(*) > 1;`}
        />

        <CodeBlock
          language="sql"
          title="problem_05_category_revenue.sql"
          code={`-- Problem 5: Revenue by category, showing % of total revenue
SELECT p.category,
       SUM(oi.quantity * oi.unit_price) AS category_revenue,
       ROUND(100.0 * SUM(oi.quantity * oi.unit_price)
             / SUM(SUM(oi.quantity * oi.unit_price)) OVER (), 2) AS pct_total
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY category_revenue DESC;`}
        />

        <CodeBlock
          language="sql"
          title="problem_06_first_order.sql"
          code={`-- Problem 6: Each user's first order date and amount
SELECT DISTINCT ON (user_id)
       user_id, order_id, order_date, amount
FROM orders
WHERE status = 'completed'
ORDER BY user_id, order_date ASC;

-- Alternative (works in all SQL dialects):
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) AS rn
    FROM orders
    WHERE status = 'completed'
)
SELECT user_id, order_id, order_date, amount
FROM ranked
WHERE rn = 1;`}
        />

        <CodeBlock
          language="sql"
          title="problem_07_above_avg.sql"
          code={`-- Problem 7: Orders where amount exceeds the user's average order amount
SELECT o.*
FROM orders o
JOIN (
    SELECT user_id, AVG(amount) AS avg_amount
    FROM orders
    WHERE status = 'completed'
    GROUP BY user_id
) ua ON o.user_id = ua.user_id
WHERE o.amount > ua.avg_amount
  AND o.status = 'completed';`}
        />

        <CodeBlock
          language="sql"
          title="problem_08_consecutive_days.sql"
          code={`-- Problem 8: Users who logged in on 3+ consecutive days
WITH login_days AS (
    SELECT DISTINCT user_id, login_date
    FROM logins
),
grouped AS (
    SELECT user_id,
           login_date,
           login_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date)
               * INTERVAL '1 day' AS grp
    FROM login_days
)
SELECT user_id,
       MIN(login_date) AS streak_start,
       MAX(login_date) AS streak_end,
       COUNT(*) AS streak_length
FROM grouped
GROUP BY user_id, grp
HAVING COUNT(*) >= 3
ORDER BY streak_length DESC;`}
        />

        <h3>Medium: Window Functions and Subqueries (Problems 9-18)</h3>

        <CodeBlock
          language="sql"
          title="problem_09_running_total.sql"
          code={`-- Problem 9: Running total of revenue per user, ordered by date
SELECT user_id,
       order_date,
       amount,
       SUM(amount) OVER (
           PARTITION BY user_id
           ORDER BY order_date
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running_total
FROM orders
WHERE status = 'completed'
ORDER BY user_id, order_date;`}
        />

        <CodeBlock
          language="sql"
          title="problem_10_rank_in_dept.sql"
          code={`-- Problem 10: Top 3 earners in each department
WITH ranked AS (
    SELECT emp_id, name, department, salary,
           DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
    FROM employees
)
SELECT emp_id, name, department, salary, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY department, rnk;`}
        />

        <CodeBlock
          language="sql"
          title="problem_11_median_salary.sql"
          code={`-- Problem 11: Median salary per department
SELECT department,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees
GROUP BY department;

-- If PERCENTILE_CONT not available, use the manual approach:
WITH ranked AS (
    SELECT department, salary,
           ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary) AS rn,
           COUNT(*) OVER (PARTITION BY department) AS cnt
    FROM employees
)
SELECT department,
       AVG(salary) AS median_salary
FROM ranked
WHERE rn IN (FLOOR((cnt + 1) / 2.0), CEIL((cnt + 1) / 2.0))
GROUP BY department;`}
        />

        <CodeBlock
          language="sql"
          title="problem_12_yoy_growth.sql"
          code={`-- Problem 12: Year-over-year revenue growth by category
WITH yearly AS (
    SELECT p.category,
           EXTRACT(YEAR FROM o.order_date) AS yr,
           SUM(oi.quantity * oi.unit_price) AS revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.status = 'completed'
    GROUP BY p.category, EXTRACT(YEAR FROM o.order_date)
)
SELECT category, yr, revenue,
       LAG(revenue) OVER (PARTITION BY category ORDER BY yr) AS prev_year_rev,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (PARTITION BY category ORDER BY yr))
             / NULLIF(LAG(revenue) OVER (PARTITION BY category ORDER BY yr), 0), 2)
           AS yoy_growth_pct
FROM yearly
ORDER BY category, yr;`}
        />

        <CodeBlock
          language="sql"
          title="problem_13_retention.sql"
          code={`-- Problem 13: Monthly retention rate (% of users who ordered in month M
-- who also ordered in month M+1)
WITH user_months AS (
    SELECT DISTINCT user_id,
           DATE_TRUNC('month', order_date) AS month
    FROM orders
    WHERE status = 'completed'
)
SELECT a.month,
       COUNT(DISTINCT a.user_id) AS active_users,
       COUNT(DISTINCT b.user_id) AS retained_users,
       ROUND(100.0 * COUNT(DISTINCT b.user_id)
             / NULLIF(COUNT(DISTINCT a.user_id), 0), 2) AS retention_pct
FROM user_months a
LEFT JOIN user_months b
    ON a.user_id = b.user_id
    AND b.month = a.month + INTERVAL '1 month'
GROUP BY a.month
ORDER BY a.month;`}
        />

        <CodeBlock
          language="sql"
          title="problem_14_gaps_islands.sql"
          code={`-- Problem 14: Find gaps in sequential order IDs
WITH id_seq AS (
    SELECT order_id,
           LEAD(order_id) OVER (ORDER BY order_id) AS next_id
    FROM orders
)
SELECT order_id + 1 AS gap_start,
       next_id - 1 AS gap_end,
       next_id - order_id - 1 AS gap_size
FROM id_seq
WHERE next_id - order_id > 1
ORDER BY gap_start;`}
        />

        <CodeBlock
          language="sql"
          title="problem_15_moving_avg.sql"
          code={`-- Problem 15: 7-day moving average of daily revenue
WITH daily AS (
    SELECT order_date,
           SUM(amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY order_date
)
SELECT order_date,
       daily_revenue,
       AVG(daily_revenue) OVER (
           ORDER BY order_date
           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS moving_avg_7d
FROM daily
ORDER BY order_date;`}
        />

        <CodeBlock
          language="sql"
          title="problem_16_manager_hierarchy.sql"
          code={`-- Problem 16: Employee hierarchy — find all reports (direct + indirect) for a manager
WITH RECURSIVE reports AS (
    -- Base: direct reports
    SELECT emp_id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id = 1  -- Manager ID = 1

    UNION ALL

    -- Recursive: indirect reports
    SELECT e.emp_id, e.name, e.manager_id, r.level + 1
    FROM employees e
    JOIN reports r ON e.manager_id = r.emp_id
)
SELECT emp_id, name, level
FROM reports
ORDER BY level, name;`}
        />

        <CodeBlock
          language="sql"
          title="problem_17_sessionization.sql"
          code={`-- Problem 17: Sessionize user activity — new session if gap > 30 minutes
WITH login_ordered AS (
    SELECT user_id, login_date, session_duration_min,
           LAG(login_date) OVER (PARTITION BY user_id ORDER BY login_date) AS prev_login
    FROM logins
),
flagged AS (
    SELECT *,
           CASE WHEN prev_login IS NULL
                     OR login_date - prev_login > INTERVAL '30 minutes'
                THEN 1 ELSE 0
           END AS new_session_flag
    FROM login_ordered
)
SELECT user_id, login_date,
       SUM(new_session_flag) OVER (
           PARTITION BY user_id ORDER BY login_date
       ) AS session_id
FROM flagged;`}
        />

        <CodeBlock
          language="sql"
          title="problem_18_funnel.sql"
          code={`-- Problem 18: Conversion funnel — signup -> first_login -> first_order -> second_order
WITH first_login AS (
    SELECT user_id, MIN(login_date) AS first_login_date
    FROM logins GROUP BY user_id
),
first_order AS (
    SELECT user_id, MIN(order_date) AS first_order_date
    FROM orders WHERE status = 'completed' GROUP BY user_id
),
second_order AS (
    SELECT user_id, order_date AS second_order_date
    FROM (
        SELECT user_id, order_date,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) AS rn
        FROM orders WHERE status = 'completed'
    ) t WHERE rn = 2
)
SELECT
    COUNT(DISTINCT u.user_id) AS signups,
    COUNT(DISTINCT fl.user_id) AS logged_in,
    COUNT(DISTINCT fo.user_id) AS first_purchase,
    COUNT(DISTINCT so.user_id) AS repeat_purchase,
    ROUND(100.0 * COUNT(DISTINCT fl.user_id) / COUNT(DISTINCT u.user_id), 1) AS login_rate,
    ROUND(100.0 * COUNT(DISTINCT fo.user_id) / NULLIF(COUNT(DISTINCT fl.user_id), 0), 1) AS purchase_rate,
    ROUND(100.0 * COUNT(DISTINCT so.user_id) / NULLIF(COUNT(DISTINCT fo.user_id), 0), 1) AS repeat_rate
FROM users u
LEFT JOIN first_login fl ON u.user_id = fl.user_id
LEFT JOIN first_order fo ON u.user_id = fo.user_id
LEFT JOIN second_order so ON u.user_id = so.user_id;`}
        />

        <h3>Hard: Advanced Patterns (Problems 19-30)</h3>

        <CodeBlock
          language="sql"
          title="problem_19_pivot.sql"
          code={`-- Problem 19: Pivot — revenue by category as columns
SELECT
    DATE_TRUNC('month', o.order_date) AS month,
    SUM(CASE WHEN p.category = 'Electronics' THEN oi.quantity * oi.unit_price ELSE 0 END) AS electronics,
    SUM(CASE WHEN p.category = 'Clothing' THEN oi.quantity * oi.unit_price ELSE 0 END) AS clothing,
    SUM(CASE WHEN p.category = 'Books' THEN oi.quantity * oi.unit_price ELSE 0 END) AS books,
    SUM(oi.quantity * oi.unit_price) AS total
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.status = 'completed'
GROUP BY 1
ORDER BY 1;`}
        />

        <CodeBlock
          language="sql"
          title="problem_20_cumulative_dist.sql"
          code={`-- Problem 20: What % of users account for 80% of revenue? (Pareto analysis)
WITH user_revenue AS (
    SELECT user_id, SUM(amount) AS total_spent
    FROM orders WHERE status = 'completed'
    GROUP BY user_id
),
ranked AS (
    SELECT user_id, total_spent,
           SUM(total_spent) OVER (ORDER BY total_spent DESC) AS cumulative_rev,
           SUM(total_spent) OVER () AS grand_total,
           ROW_NUMBER() OVER (ORDER BY total_spent DESC) AS rn,
           COUNT(*) OVER () AS total_users
    FROM user_revenue
)
SELECT rn AS user_rank,
       total_spent,
       ROUND(100.0 * cumulative_rev / grand_total, 2) AS cumulative_rev_pct,
       ROUND(100.0 * rn / total_users, 2) AS cumulative_user_pct
FROM ranked
WHERE cumulative_rev / grand_total <= 0.80
   OR rn = 1  -- always include top user
ORDER BY rn;`}
        />

        <CodeBlock
          language="sql"
          title="problem_21_cohort.sql"
          code={`-- Problem 21: Cohort retention analysis — users grouped by signup month,
-- track what % are active in each subsequent month
WITH cohort AS (
    SELECT user_id,
           DATE_TRUNC('month', signup_date) AS cohort_month
    FROM users
),
activity AS (
    SELECT DISTINCT user_id,
           DATE_TRUNC('month', order_date) AS active_month
    FROM orders WHERE status = 'completed'
)
SELECT c.cohort_month,
       EXTRACT(MONTH FROM AGE(a.active_month, c.cohort_month)) AS months_since_signup,
       COUNT(DISTINCT a.user_id) AS active_users,
       COUNT(DISTINCT c.user_id) AS cohort_size,
       ROUND(100.0 * COUNT(DISTINCT a.user_id) / COUNT(DISTINCT c.user_id), 2) AS retention_pct
FROM cohort c
LEFT JOIN activity a ON c.user_id = a.user_id
    AND a.active_month >= c.cohort_month
GROUP BY c.cohort_month, EXTRACT(MONTH FROM AGE(a.active_month, c.cohort_month))
ORDER BY c.cohort_month, months_since_signup;`}
        />

        <CodeBlock
          language="sql"
          title="problem_22_dept_salary_comparison.sql"
          code={`-- Problem 22: Employees earning more than their department average
-- but less than the company average
WITH dept_avg AS (
    SELECT department, AVG(salary) AS dept_avg_salary
    FROM employees GROUP BY department
),
company_avg AS (
    SELECT AVG(salary) AS company_avg_salary FROM employees
)
SELECT e.emp_id, e.name, e.department, e.salary,
       d.dept_avg_salary, c.company_avg_salary
FROM employees e
JOIN dept_avg d ON e.department = d.department
CROSS JOIN company_avg c
WHERE e.salary > d.dept_avg_salary
  AND e.salary < c.company_avg_salary;`}
        />

        <CodeBlock
          language="sql"
          title="problem_23_market_basket.sql"
          code={`-- Problem 23: Market basket analysis — products frequently bought together
SELECT a.product_id AS product_a,
       b.product_id AS product_b,
       COUNT(DISTINCT a.order_id) AS co_occurrence
FROM order_items a
JOIN order_items b
    ON a.order_id = b.order_id
    AND a.product_id < b.product_id  -- Avoid duplicates and self-joins
GROUP BY a.product_id, b.product_id
HAVING COUNT(DISTINCT a.order_id) >= 5
ORDER BY co_occurrence DESC
LIMIT 20;`}
        />

        <CodeBlock
          language="sql"
          title="problem_24_time_between.sql"
          code={`-- Problem 24: Average time between first and second purchase
WITH ordered AS (
    SELECT user_id, order_date,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) AS order_num
    FROM orders
    WHERE status = 'completed'
)
SELECT
    AVG(second.order_date - first.order_date) AS avg_days_to_second_order,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY second.order_date - first.order_date
    ) AS median_days_to_second_order
FROM ordered first
JOIN ordered second
    ON first.user_id = second.user_id
WHERE first.order_num = 1
  AND second.order_num = 2;`}
        />

        <CodeBlock
          language="sql"
          title="problem_25_dedup_keep_latest.sql"
          code={`-- Problem 25: Deduplicate — keep only the most recent order per user per day
DELETE FROM orders
WHERE order_id NOT IN (
    SELECT DISTINCT ON (user_id, order_date) order_id
    FROM orders
    ORDER BY user_id, order_date, order_id DESC
);

-- Non-destructive version (SELECT):
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY user_id, order_date
               ORDER BY order_id DESC
           ) AS rn
    FROM orders
)
SELECT * FROM ranked WHERE rn = 1;`}
        />

        <CodeBlock
          language="sql"
          title="problem_26_rolling_active.sql"
          code={`-- Problem 26: 28-day rolling active users (DAU, WAU, MAU)
WITH daily_active AS (
    SELECT DISTINCT login_date, user_id
    FROM logins
)
SELECT login_date,
       COUNT(DISTINCT user_id) AS dau,
       COUNT(DISTINCT user_id) OVER (
           ORDER BY login_date
           RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
       ) AS wau_7d,
       COUNT(DISTINCT user_id) OVER (
           ORDER BY login_date
           RANGE BETWEEN INTERVAL '27 days' PRECEDING AND CURRENT ROW
       ) AS mau_28d
FROM daily_active
GROUP BY login_date
ORDER BY login_date;`}
        />

        <CodeBlock
          language="sql"
          title="problem_27_attribution.sql"
          code={`-- Problem 27: Last-touch attribution — which channel gets credit for each purchase
WITH touchpoints AS (
    -- Assume a 'marketing_touches' table with channel info
    SELECT user_id, touch_date, channel,
           ROW_NUMBER() OVER (
               PARTITION BY user_id
               ORDER BY touch_date DESC
           ) AS rn
    FROM marketing_touches
)
SELECT t.channel,
       COUNT(DISTINCT o.order_id) AS attributed_orders,
       SUM(o.amount) AS attributed_revenue
FROM orders o
JOIN touchpoints t ON o.user_id = t.user_id
    AND t.touch_date <= o.order_date
    AND t.rn = 1  -- Last touch before purchase
WHERE o.status = 'completed'
GROUP BY t.channel
ORDER BY attributed_revenue DESC;`}
        />

        <CodeBlock
          language="sql"
          title="problem_28_ntile_segmentation.sql"
          code={`-- Problem 28: RFM segmentation using NTILE
WITH rfm AS (
    SELECT user_id,
           MAX(order_date) AS last_order,
           COUNT(*) AS frequency,
           SUM(amount) AS monetary
    FROM orders
    WHERE status = 'completed'
    GROUP BY user_id
)
SELECT user_id,
       NTILE(5) OVER (ORDER BY last_order DESC) AS recency_score,
       NTILE(5) OVER (ORDER BY frequency ASC) AS frequency_score,
       NTILE(5) OVER (ORDER BY monetary ASC) AS monetary_score
FROM rfm;`}
        />

        <CodeBlock
          language="sql"
          title="problem_29_event_sequence.sql"
          code={`-- Problem 29: Find users who did action A then action B within 1 hour
-- (assumes an events table with event_type and event_time)
WITH event_pairs AS (
    SELECT a.user_id,
           a.event_time AS action_a_time,
           b.event_time AS action_b_time
    FROM events a
    JOIN events b ON a.user_id = b.user_id
        AND b.event_type = 'B'
        AND a.event_type = 'A'
        AND b.event_time > a.event_time
        AND b.event_time <= a.event_time + INTERVAL '1 hour'
)
SELECT DISTINCT user_id
FROM event_pairs;`}
        />

        <CodeBlock
          language="sql"
          title="problem_30_largest_gap.sql"
          code={`-- Problem 30: For each user, find the largest gap (in days) between consecutive orders
WITH order_gaps AS (
    SELECT user_id,
           order_date,
           LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date) AS prev_date,
           order_date - LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date) AS gap_days
    FROM orders
    WHERE status = 'completed'
)
SELECT user_id,
       MAX(gap_days) AS largest_gap_days
FROM order_gaps
WHERE gap_days IS NOT NULL
GROUP BY user_id
ORDER BY largest_gap_days DESC;`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always use CTEs for readability</strong>: Nested subqueries are hard to debug. CTEs (WITH clauses) make your logic modular and easy to test step-by-step.</li>
          <li><strong>Use NULLIF to avoid division by zero</strong>: <code>x / NULLIF(y, 0)</code> returns NULL instead of an error when the denominator is zero.</li>
          <li><strong>DISTINCT ON (PostgreSQL) is a lifesaver</strong>: For &quot;first/last per group&quot; queries, it is cleaner than ROW_NUMBER + filter. In other dialects, use ROW_NUMBER.</li>
          <li><strong>Think about NULLs</strong>: NULL != NULL, NULL is not equal to anything, and aggregate functions ignore NULLs (except COUNT(*)). Always consider how NULLs affect your results.</li>
          <li><strong>Test with edge cases</strong>: Empty groups, single-row groups, ties, NULLs, and boundary dates. Interviewers love to ask &quot;what happens if...&quot;</li>
          <li><strong>Explain your approach before writing</strong>: In an interview, narrate your thought process: &quot;I need to group by user, compute running totals, then filter. I will use a CTE with a window function.&quot;</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>GROUP BY and non-aggregated columns</strong>: Every column in SELECT must be either in GROUP BY or inside an aggregate function. This is the most common SQL error.</li>
          <li><strong>Using RANK when you want DENSE_RANK</strong>: RANK skips numbers after ties (1,1,3). DENSE_RANK does not (1,1,2). ROW_NUMBER never ties (1,2,3). Know which one you need.</li>
          <li><strong>Forgetting LEFT JOIN NULLs</strong>: After a LEFT JOIN, columns from the right table are NULL for non-matching rows. Filtering on a right-table column in WHERE (not IS NULL) silently converts it to an INNER JOIN.</li>
          <li><strong>Window function in WHERE</strong>: You cannot use window functions in WHERE because WHERE executes before SELECT. Wrap in a CTE or subquery first.</li>
          <li><strong>COUNT(column) vs COUNT(*)</strong>: COUNT(column) excludes NULLs. COUNT(*) counts all rows. Use the right one.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Bonus Challenge:</strong> Write a query to find the longest streak of consecutive days each user placed an order.</p>
        <p><strong>Solution:</strong></p>
        <CodeBlock
          language="sql"
          title="bonus_longest_streak.sql"
          code={`-- Classic "gaps and islands" problem
WITH distinct_days AS (
    SELECT DISTINCT user_id, order_date
    FROM orders
    WHERE status = 'completed'
),
islands AS (
    SELECT user_id,
           order_date,
           order_date - (ROW_NUMBER() OVER (
               PARTITION BY user_id ORDER BY order_date
           ))::int * INTERVAL '1 day' AS island_id
    FROM distinct_days
)
SELECT user_id,
       MIN(order_date) AS streak_start,
       MAX(order_date) AS streak_end,
       COUNT(*) AS streak_length
FROM islands
GROUP BY user_id, island_id
ORDER BY streak_length DESC;

-- How it works:
-- If dates are consecutive: 2024-01-01, 2024-01-02, 2024-01-03
-- Minus row_number (1,2,3) gives: 2024-01-00, 2024-01-00, 2024-01-00
-- Same "island_id" = same streak!`}
        />
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>LeetCode SQL section</strong> — 50+ problems sorted by difficulty, great for timed practice.</li>
          <li><strong>StrataScratch</strong> — Real interview SQL questions from FAANG companies with discussion threads.</li>
          <li><strong>&quot;SQL for Data Scientists&quot; by Renee Teate</strong> — From basics to advanced analytics patterns.</li>
          <li><strong>Mode Analytics SQL tutorial</strong> — Interactive environment with real datasets and window function deep dives.</li>
          <li><strong>Use the Index, Luke (use-the-index-luke.com)</strong> — SQL performance and indexing guide. Critical for production SQL.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
