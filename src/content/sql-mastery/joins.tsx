"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Joins() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Real-world data lives in multiple tables. Users are in one table, their orders in another, products in a third.
          <strong> JOINs</strong> are how you combine these tables back together to answer meaningful questions.
          A JOIN matches rows from two tables based on a condition — usually matching a foreign key to a primary key.
        </p>
        <p>
          The most common join is <strong>INNER JOIN</strong>, which returns only rows that have a match in
          both tables. If a user has no orders, they disappear from the result. A <strong>LEFT JOIN</strong> keeps
          all rows from the left table even if there is no match — unmatched columns from the right table become NULL.
          This is essential for questions like &quot;show me all users AND their orders, including users who have never
          ordered.&quot;
        </p>
        <p>
          <strong>RIGHT JOIN</strong> is the mirror of LEFT JOIN — it keeps all rows from the right table.
          In practice, most people rewrite RIGHT JOINs as LEFT JOINs by swapping table order, since it reads more
          naturally. A <strong>FULL OUTER JOIN</strong> keeps all rows from both tables, filling in NULLs on
          whichever side lacks a match. It is useful for reconciliation tasks — finding records that exist in one
          system but not another.
        </p>
        <p>
          Two specialized joins deserve attention. A <strong>CROSS JOIN</strong> produces the Cartesian product
          of two tables — every row paired with every other row. This is rarely used on large tables but is powerful
          for generating combinations (e.g., all date-product pairs for filling in zero-sales days).
          A <strong>SELF JOIN</strong> joins a table to itself, which is useful for comparing rows within the same
          table — such as finding employees and their managers, or orders placed on consecutive days.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Set Theory of JOINs</h3>
        <p>
          If table <InlineMath math="A" /> has <InlineMath math="m" /> rows and table <InlineMath math="B" /> has <InlineMath math="n" /> rows:
        </p>

        <p><strong>INNER JOIN</strong> — the intersection of matching rows:</p>
        <BlockMath math="A \bowtie_{A.key = B.key} B = \{(a, b) \mid a \in A, \; b \in B, \; a.key = b.key\}" />

        <p><strong>LEFT JOIN</strong> — all of A, plus matching rows from B (NULLs where no match):</p>
        <BlockMath math="A \; ⟕ \; B = (A \bowtie B) \cup \{(a, \text{NULL}) \mid a \in A, \; \nexists \; b \in B : a.key = b.key\}" />

        <p><strong>FULL OUTER JOIN</strong> — all rows from both sides:</p>
        <BlockMath math="A \; ⟗ \; B = (A \bowtie B) \cup (\text{unmatched from } A) \cup (\text{unmatched from } B)" />

        <p><strong>CROSS JOIN</strong> — the Cartesian product:</p>
        <BlockMath math="A \times B = \{(a, b) \mid a \in A, \; b \in B\}, \quad |A \times B| = m \cdot n" />

        <h3>Cardinality</h3>
        <p>
          The output size of a join depends on the key relationship. For a <strong>one-to-many</strong> join
          (e.g., users to orders), the result has at most <InlineMath math="n" /> rows (one per order).
          For a <strong>many-to-many</strong> join, the result can explode — if 5 rows in A match 3 rows in B
          for the same key, that key contributes <InlineMath math="5 \times 3 = 15" /> rows. This is the most
          common source of unexpected row count changes in queries.
        </p>
        <BlockMath math="|A \bowtie B| \leq m \cdot n \quad \text{(worst case: every row matches every row)}" />
      </TopicSection>

      <TopicSection type="code">
        <h3>Schema Setup</h3>
        <CodeBlock
          language="sql"
          title="schema.sql"
          code={`-- Users table
-- users (user_id PK, name, email, signup_date, country)

-- Orders table
-- orders (order_id PK, user_id FK, product_id FK, quantity, unit_price, order_date, status)

-- Products table
-- products (product_id PK, product_name, category, list_price)`}
        />

        <h3>INNER JOIN — Only Matching Rows</h3>
        <CodeBlock
          language="sql"
          title="inner_join.sql"
          code={`-- Users and their completed orders (excludes users with no orders)
SELECT
    u.user_id,
    u.name,
    o.order_id,
    o.order_date,
    o.quantity * o.unit_price AS order_total
FROM users u
INNER JOIN orders o
    ON u.user_id = o.user_id
WHERE o.status = 'completed'
ORDER BY o.order_date DESC;`}
        />

        <h3>LEFT JOIN — Keep All Left Rows</h3>
        <CodeBlock
          language="sql"
          title="left_join.sql"
          code={`-- ALL users and their order count (including users with zero orders)
SELECT
    u.user_id,
    u.name,
    u.signup_date,
    COUNT(o.order_id)                   AS order_count,
    COALESCE(SUM(o.quantity * o.unit_price), 0) AS total_spend
FROM users u
LEFT JOIN orders o
    ON u.user_id = o.user_id
   AND o.status = 'completed'        -- filter goes in ON, not WHERE
GROUP BY u.user_id, u.name, u.signup_date
ORDER BY total_spend DESC;

-- Find users who have NEVER placed an order
SELECT u.user_id, u.name, u.signup_date
FROM users u
LEFT JOIN orders o
    ON u.user_id = o.user_id
WHERE o.order_id IS NULL;             -- anti-join pattern`}
        />

        <h3>FULL OUTER JOIN — Reconciliation</h3>
        <CodeBlock
          language="sql"
          title="full_outer_join.sql"
          code={`-- Reconcile two payment systems: find mismatches
-- payments_stripe (txn_id, amount, created_at)
-- payments_internal (txn_id, amount, processed_at)

SELECT
    COALESCE(s.txn_id, i.txn_id)   AS txn_id,
    s.amount                         AS stripe_amount,
    i.amount                         AS internal_amount,
    CASE
        WHEN s.txn_id IS NULL     THEN 'missing_in_stripe'
        WHEN i.txn_id IS NULL     THEN 'missing_in_internal'
        WHEN s.amount != i.amount THEN 'amount_mismatch'
        ELSE 'matched'
    END AS reconciliation_status
FROM payments_stripe s
FULL OUTER JOIN payments_internal i
    ON s.txn_id = i.txn_id
WHERE s.txn_id IS NULL
   OR i.txn_id IS NULL
   OR s.amount != i.amount
ORDER BY txn_id;`}
        />

        <h3>CROSS JOIN — Generate All Combinations</h3>
        <CodeBlock
          language="sql"
          title="cross_join.sql"
          code={`-- Generate a row for every date x product combination,
-- then fill in actual sales (useful for "zero-fill" reports)
WITH date_range AS (
    SELECT generate_series(
        '2025-01-01'::date,
        '2025-01-31'::date,
        '1 day'::interval
    )::date AS sale_date
),
all_combos AS (
    SELECT d.sale_date, p.product_id, p.product_name
    FROM date_range d
    CROSS JOIN products p
)
SELECT
    ac.sale_date,
    ac.product_name,
    COALESCE(SUM(o.quantity), 0) AS units_sold
FROM all_combos ac
LEFT JOIN orders o
    ON o.product_id = ac.product_id
   AND DATE(o.order_date) = ac.sale_date
   AND o.status = 'completed'
GROUP BY ac.sale_date, ac.product_name
ORDER BY ac.sale_date, ac.product_name;`}
        />

        <h3>SELF JOIN — Comparing Rows Within the Same Table</h3>
        <CodeBlock
          language="sql"
          title="self_join.sql"
          code={`-- Table: employees (emp_id, name, manager_id, department, salary)

-- Find each employee and their manager's name
SELECT
    e.emp_id,
    e.name       AS employee_name,
    e.department,
    m.name       AS manager_name
FROM employees e
LEFT JOIN employees m
    ON e.manager_id = m.emp_id
ORDER BY e.department, e.name;

-- Find employees who earn more than their manager
SELECT
    e.name       AS employee_name,
    e.salary     AS employee_salary,
    m.name       AS manager_name,
    m.salary     AS manager_salary
FROM employees e
INNER JOIN employees m
    ON e.manager_id = m.emp_id
WHERE e.salary > m.salary;`}
        />

        <h3>Multi-Table JOIN</h3>
        <CodeBlock
          language="sql"
          title="multi_table_join.sql"
          code={`-- Full order details: user name, product name, totals
SELECT
    u.name                          AS customer_name,
    p.product_name,
    p.category,
    o.quantity,
    o.unit_price,
    o.quantity * o.unit_price       AS line_total,
    o.order_date
FROM orders o
INNER JOIN users u     ON o.user_id    = u.user_id
INNER JOIN products p  ON o.product_id = p.product_id
WHERE o.status = 'completed'
  AND o.order_date >= '2025-01-01'
ORDER BY o.order_date DESC, line_total DESC;`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>LEFT JOIN + IS NULL is the anti-join pattern</strong>: To find rows in A with no match in B,
            use LEFT JOIN ... WHERE b.key IS NULL. This is often more readable (and sometimes faster) than
            NOT EXISTS or NOT IN.
          </li>
          <li>
            <strong>Put filters in ON vs WHERE carefully</strong>: For INNER JOINs, it does not matter. For LEFT JOINs,
            it matters enormously. A filter in WHERE on the right table effectively converts a LEFT JOIN into an
            INNER JOIN because NULLs fail the WHERE check.
          </li>
          <li>
            <strong>Always check row counts after a JOIN</strong>: If your row count unexpectedly increases, you likely
            have a many-to-many relationship. Add a DISTINCT or fix your join keys.
          </li>
          <li>
            <strong>Index your join columns</strong>: JOINs on non-indexed columns force nested-loop scans.
            Ensure foreign keys and join columns have indexes. The difference on large tables is orders of magnitude.
          </li>
          <li>
            <strong>Prefer explicit JOIN syntax over implicit</strong>: Use FROM a JOIN b ON ... rather than
            FROM a, b WHERE a.key = b.key. The explicit syntax is clearer, harder to accidentally create a
            Cartesian product, and has been the standard since SQL-92.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Accidental Cartesian product</strong>: Forgetting the ON clause or joining on the wrong columns
            produces a CROSS JOIN. If table A has 10K rows and table B has 50K rows, you get 500 million rows.
            Always verify your ON conditions.
          </li>
          <li>
            <strong>Filtering a LEFT JOIN in WHERE instead of ON</strong>: Writing LEFT JOIN orders o ON u.user_id = o.user_id
            WHERE o.status = &apos;completed&apos; eliminates all users with no orders (because o.status is NULL for them and
            NULL != &apos;completed&apos;). Move the filter to the ON clause to preserve unmatched left rows.
          </li>
          <li>
            <strong>Using NOT IN with NULLs</strong>: NOT IN (subquery) returns no rows if the subquery contains any
            NULL value. Use NOT EXISTS or LEFT JOIN + IS NULL instead — they handle NULLs correctly.
          </li>
          <li>
            <strong>Many-to-many join explosion</strong>: Joining orders to order_items (1-to-many) and then to
            promotions (many-to-many) can silently multiply row counts. Aggregate to the correct grain before joining
            or use DISTINCT to de-duplicate.
          </li>
          <li>
            <strong>Assuming RIGHT JOIN is necessary</strong>: In 99% of cases, rewriting as a LEFT JOIN by swapping
            table order is more readable. Most style guides discourage RIGHT JOINs.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Given tables <code>users(user_id, name, signup_date)</code> and
          <code> orders(order_id, user_id, amount, order_date)</code>, write a query that returns all users who signed up
          in 2025 but have never placed an order. Then extend it to also show users who signed up in 2025 and
          placed their first order more than 30 days after signup.
        </p>
        <CodeBlock
          language="sql"
          title="interview_solution.sql"
          code={`-- Part 1: Users who signed up in 2025 but never ordered
SELECT u.user_id, u.name, u.signup_date
FROM users u
LEFT JOIN orders o
    ON u.user_id = o.user_id
WHERE u.signup_date >= '2025-01-01'
  AND u.signup_date <  '2026-01-01'
  AND o.order_id IS NULL;

-- Part 2: Also include users whose first order was > 30 days after signup
SELECT
    u.user_id,
    u.name,
    u.signup_date,
    MIN(o.order_date) AS first_order_date,
    CASE
        WHEN MIN(o.order_date) IS NULL THEN 'never_ordered'
        ELSE 'slow_activator'
    END AS segment
FROM users u
LEFT JOIN orders o
    ON u.user_id = o.user_id
WHERE u.signup_date >= '2025-01-01'
  AND u.signup_date <  '2026-01-01'
GROUP BY u.user_id, u.name, u.signup_date
HAVING MIN(o.order_date) IS NULL
    OR MIN(o.order_date) - u.signup_date > 30
ORDER BY u.signup_date;`}
        />
        <p>
          <strong>Key insight:</strong> Part 1 tests the LEFT JOIN + IS NULL anti-join pattern. Part 2 tests
          whether you can combine HAVING with aggregated vs non-aggregated conditions and handle the NULL case
          (never ordered) alongside the date arithmetic case (slow activators) in a single query.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li>
            <strong>&quot;SQL Joins Visualizer&quot; (sql-joins.leopard.in.ua)</strong> — Interactive Venn diagram tool
            that lets you see exactly which rows survive each join type.
          </li>
          <li>
            <strong>PostgreSQL EXPLAIN documentation</strong> — Learn to read execution plans and understand
            whether your joins use Hash Join, Merge Join, or Nested Loop strategies.
          </li>
          <li>
            <strong>&quot;SQL Performance Explained&quot; Chapter 4: The Join Operation</strong> — Deep dive into
            how the database engine physically executes each join algorithm.
          </li>
          <li>
            <strong>LeetCode SQL problems (Medium/Hard)</strong> — Problems #175 (Combine Two Tables),
            #181 (Employees Earning More Than Their Managers), and #183 (Customers Who Never Order) are
            classic join exercises.
          </li>
        </ul>
      </TopicSection>
    </div>
  );
}
