"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function SQLFundamentals() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          SQL (Structured Query Language) is the universal language for talking to relational databases. Every time you
          pull a report, build a dashboard, or feed data into a machine learning pipeline, SQL is almost certainly
          involved. It&apos;s a <strong>declarative</strong> language — you describe <em>what</em> you want, and the database
          engine figures out <em>how</em> to get it.
        </p>
        <p>
          The core of SQL is the <strong>SELECT</strong> statement. You choose which columns to return, which table to
          read from, which rows to keep (WHERE), how to group them (GROUP BY), how to filter groups (HAVING), and how to
          sort the results (ORDER BY). These six clauses — SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY — form the
          backbone of virtually every analytical query you will ever write.
        </p>
        <p>
          It is critical to understand the <strong>logical order of execution</strong>, which differs from the order you
          write the clauses. The database processes FROM first (which table?), then WHERE (filter rows), then GROUP BY
          (aggregate), then HAVING (filter groups), then SELECT (pick columns), and finally ORDER BY (sort output).
          Understanding this order explains why you cannot reference a column alias from SELECT inside a WHERE clause — the
          WHERE runs before SELECT does.
        </p>
        <p>
          SQL operates on <strong>sets of rows</strong>, not individual records. Thinking in sets rather than loops is the
          single most important mental shift for writing efficient queries. Instead of &quot;loop through each order and check
          if the amount is above 100,&quot; you think &quot;give me the set of orders where amount is above 100.&quot;
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Relational Algebra Foundations</h3>
        <p>
          SQL is grounded in <strong>relational algebra</strong>. Each SQL clause maps to a relational algebra operation:
        </p>
        <p>
          <strong>Selection (WHERE)</strong> — filters rows based on a predicate <InlineMath math="\sigma" />:
        </p>
        <BlockMath math="\sigma_{price > 100}(Orders)" />
        <p>
          <strong>Projection (SELECT)</strong> — picks a subset of columns <InlineMath math="\pi" />:
        </p>
        <BlockMath math="\pi_{user\_id, total}(Orders)" />
        <p>
          <strong>Aggregation (GROUP BY)</strong> — partitions a relation and applies aggregate functions <InlineMath math="\mathcal{G}" />:
        </p>
        <BlockMath math="_{user\_id}\mathcal{G}_{SUM(total), COUNT(*)}(Orders)" />

        <h3>Logical Order of Execution</h3>
        <BlockMath math="\text{FROM} \rightarrow \text{WHERE} \rightarrow \text{GROUP BY} \rightarrow \text{HAVING} \rightarrow \text{SELECT} \rightarrow \text{ORDER BY} \rightarrow \text{LIMIT}" />

        <h3>Cardinality and Complexity</h3>
        <p>
          A full table scan of <InlineMath math="n" /> rows is <InlineMath math="O(n)" />. With a B-tree index on the
          WHERE column, lookup becomes <InlineMath math="O(\log n)" />. Sorting for ORDER BY is
          typically <InlineMath math="O(n \log n)" />. GROUP BY with a hash aggregate is <InlineMath math="O(n)" /> on
          average. Understanding these complexities helps you predict query performance on large tables.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Basic SELECT with Filtering</h3>
        <CodeBlock
          language="sql"
          title="basic_select.sql"
          code={`-- Table: orders (order_id, user_id, product_id, quantity, unit_price, order_date, status)

-- 1. Simple SELECT with WHERE
SELECT order_id, user_id, quantity * unit_price AS total_amount
FROM orders
WHERE status = 'completed'
  AND order_date >= '2025-01-01'
ORDER BY total_amount DESC;

-- 2. Using IN, BETWEEN, LIKE for flexible filtering
SELECT order_id, user_id, order_date
FROM orders
WHERE status IN ('completed', 'shipped')
  AND unit_price BETWEEN 10.00 AND 500.00
  AND user_id IS NOT NULL
ORDER BY order_date;`}
        />

        <h3>GROUP BY and Aggregations</h3>
        <CodeBlock
          language="sql"
          title="group_by.sql"
          code={`-- Table: orders (order_id, user_id, product_id, quantity, unit_price, order_date, status)

-- Revenue per user, only users with > $1000 total spend
SELECT
    user_id,
    COUNT(*)                          AS num_orders,
    SUM(quantity * unit_price)        AS total_revenue,
    AVG(quantity * unit_price)        AS avg_order_value,
    MIN(order_date)                   AS first_order,
    MAX(order_date)                   AS last_order
FROM orders
WHERE status = 'completed'
GROUP BY user_id
HAVING SUM(quantity * unit_price) > 1000
ORDER BY total_revenue DESC;`}
        />

        <h3>Multiple Aggregation Levels</h3>
        <CodeBlock
          language="sql"
          title="multi_level_agg.sql"
          code={`-- Table: events (event_id, user_id, event_type, page_url, created_at)

-- Daily active users and events per day
SELECT
    DATE(created_at)       AS event_date,
    COUNT(DISTINCT user_id) AS daily_active_users,
    COUNT(*)                AS total_events,
    ROUND(
        COUNT(*) * 1.0 / COUNT(DISTINCT user_id), 2
    )                       AS events_per_user
FROM events
WHERE created_at >= '2025-01-01'
GROUP BY DATE(created_at)
HAVING COUNT(DISTINCT user_id) >= 100
ORDER BY event_date;`}
        />

        <h3>CASE Expressions for Conditional Logic</h3>
        <CodeBlock
          language="sql"
          title="case_expressions.sql"
          code={`-- Segment users by spend tier
SELECT
    user_id,
    SUM(quantity * unit_price) AS total_spend,
    CASE
        WHEN SUM(quantity * unit_price) >= 5000 THEN 'whale'
        WHEN SUM(quantity * unit_price) >= 1000 THEN 'high_value'
        WHEN SUM(quantity * unit_price) >= 200  THEN 'medium'
        ELSE 'low'
    END AS spend_tier,
    COUNT(*) AS order_count
FROM orders
WHERE status = 'completed'
GROUP BY user_id
ORDER BY total_spend DESC;`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>Always filter early</strong>: Place conditions in WHERE rather than HAVING when possible.
            WHERE filters individual rows before aggregation, which reduces the amount of data the database must process.
            HAVING filters after aggregation and should only be used for conditions on aggregate results.
          </li>
          <li>
            <strong>Use COUNT(DISTINCT col) carefully</strong>: It is more expensive than COUNT(*) because the engine must
            track unique values. On very large tables, consider approximate alternatives like HyperLogLog
            (e.g., APPROX_COUNT_DISTINCT in BigQuery).
          </li>
          <li>
            <strong>Avoid SELECT *</strong>: In production queries, always list specific columns. SELECT * reads every
            column from disk, wastes I/O, and breaks downstream code if the schema changes.
          </li>
          <li>
            <strong>Index your WHERE and JOIN columns</strong>: The single biggest performance lever. A query filtering on
            order_date with no index scans the entire table. Adding an index on order_date makes it nearly instant.
          </li>
          <li>
            <strong>NULLs propagate</strong>: Any arithmetic or comparison with NULL returns NULL, not false.
            Use IS NULL / IS NOT NULL, and COALESCE() to provide defaults.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Referencing an alias in WHERE</strong>: Writing
            WHERE total_amount &gt; 100 when total_amount is a SELECT alias fails because WHERE executes before SELECT.
            Repeat the expression or use a subquery / CTE.
          </li>
          <li>
            <strong>Forgetting GROUP BY columns</strong>: Every non-aggregated column in SELECT must appear in GROUP BY.
            Some databases (MySQL in loose mode) allow it but return unpredictable results.
          </li>
          <li>
            <strong>Using HAVING instead of WHERE</strong>: Writing HAVING status = &apos;completed&apos; works but forces the
            database to aggregate first then filter. Use WHERE for row-level conditions.
          </li>
          <li>
            <strong>Implicit type coercion</strong>: Comparing a string column to an integer
            (WHERE user_id = 12345 when user_id is VARCHAR) can silently cast every row, bypassing indexes and causing
            full table scans.
          </li>
          <li>
            <strong>Ignoring NULL in aggregations</strong>: COUNT(column) skips NULLs but COUNT(*) does not. AVG also
            ignores NULLs, which can distort your results if NULLs represent zeros.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Given a table <code>transactions(txn_id, user_id, amount, txn_date)</code>, write a
          query that returns each user&apos;s first transaction date, the amount of that first transaction, and their total
          number of transactions. Only include users who have made at least 3 transactions.
        </p>
        <CodeBlock
          language="sql"
          title="interview_solution.sql"
          code={`-- Step 1: Find each user's first transaction date
-- Step 2: Join back to get the amount of that first transaction
-- Step 3: Count total transactions per user
-- Step 4: Filter to users with >= 3 transactions

SELECT
    t.user_id,
    t.first_txn_date,
    first_txn.amount        AS first_txn_amount,
    t.total_transactions
FROM (
    SELECT
        user_id,
        MIN(txn_date)  AS first_txn_date,
        COUNT(*)       AS total_transactions
    FROM transactions
    GROUP BY user_id
    HAVING COUNT(*) >= 3
) t
JOIN transactions first_txn
  ON first_txn.user_id = t.user_id
 AND first_txn.txn_date = t.first_txn_date
ORDER BY t.total_transactions DESC, t.user_id;`}
        />
        <p>
          <strong>Key insight:</strong> You cannot directly retrieve the amount of the first transaction with a simple
          GROUP BY — you need a self-join or a window function. This question tests your understanding of the difference
          between aggregation and row-level access.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li>
            <strong>Stanford CS145 (Introduction to Databases)</strong> — Covers relational algebra,
            SQL semantics, and query execution in depth.
          </li>
          <li>
            <strong>&quot;SQL Performance Explained&quot; by Markus Winand</strong> — The best resource for understanding
            how indexes and execution plans work under the hood.
          </li>
          <li>
            <strong>use-the-index-luke.com</strong> — Free online companion to the book above, with visual
            explanations of B-tree indexes and query optimization.
          </li>
          <li>
            <strong>Mode Analytics SQL Tutorial</strong> — Interactive exercises with real datasets for practicing
            fundamentals.
          </li>
        </ul>
      </TopicSection>
    </div>
  );
}
