"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function WindowFunctions() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Window functions are the most powerful feature in SQL for analytics. They let you compute a value for
          each row based on a &quot;window&quot; of related rows — <strong>without collapsing those rows into a single
          group</strong>. This is the critical difference from GROUP BY: with GROUP BY, 100 orders become 1 row
          per user. With window functions, you still have all 100 order rows, but each one now carries additional
          computed information about its context.
        </p>
        <p>
          The syntax is <code>function() OVER (PARTITION BY ... ORDER BY ...)</code>. PARTITION BY defines which
          rows belong to the same &quot;window&quot; (like GROUP BY but without collapsing). ORDER BY defines the sequence
          within each partition. If you omit PARTITION BY, the entire result set is one partition. If you omit
          ORDER BY, there is no defined ordering within the partition (which matters for ranking and cumulative functions).
        </p>
        <p>
          The most commonly used window functions fall into three categories. <strong>Ranking functions</strong> —
          ROW_NUMBER(), RANK(), DENSE_RANK(), NTILE() — assign a position to each row within its partition.
          <strong> Offset functions</strong> — LAG(), LEAD(), FIRST_VALUE(), LAST_VALUE() — let you access values
          from other rows relative to the current row. <strong>Aggregate functions</strong> — SUM(), AVG(), COUNT(),
          MIN(), MAX() used with OVER — compute running or sliding totals without collapsing rows.
        </p>
        <p>
          Window functions execute after WHERE, GROUP BY, and HAVING but before ORDER BY and LIMIT. This means you
          can use them on already-aggregated results. They are read-only — they add computed columns but never
          filter or remove rows. To filter on a window function result, you must wrap it in a subquery or CTE.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Frame Specification</h3>
        <p>
          A window frame defines the subset of rows within a partition that feed into the function. The default
          frame for aggregate window functions with ORDER BY is:
        </p>
        <BlockMath math="\text{ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW}" />
        <p>This means a running total by default. You can customize:</p>
        <BlockMath math="\text{ROWS BETWEEN } k \text{ PRECEDING AND } k \text{ FOLLOWING}" />

        <h3>ROW_NUMBER vs RANK vs DENSE_RANK</h3>
        <p>
          Given a partition ordered by score where two rows tie at score 90:
        </p>
        <BlockMath math="\text{ROW\_NUMBER: } 1, 2, 3, 4 \quad \text{(no ties, arbitrary tiebreaker)}" />
        <BlockMath math="\text{RANK: } 1, 2, 2, 4 \quad \text{(ties share rank, next rank is skipped)}" />
        <BlockMath math="\text{DENSE\_RANK: } 1, 2, 2, 3 \quad \text{(ties share rank, no gap)}" />

        <h3>Running Aggregates</h3>
        <p>A running sum over ordered rows <InlineMath math="r_1, r_2, \ldots, r_n" />:</p>
        <BlockMath math="\text{running\_sum}(k) = \sum_{i=1}^{k} x_i" />
        <p>A moving average with window of size <InlineMath math="w" />:</p>
        <BlockMath math="\text{moving\_avg}(k) = \frac{1}{\min(k, w)} \sum_{i=\max(1,\, k-w+1)}^{k} x_i" />

        <h3>Execution Order</h3>
        <BlockMath math="\text{FROM} \to \text{WHERE} \to \text{GROUP BY} \to \text{HAVING} \to \text{Window Functions} \to \text{SELECT} \to \text{ORDER BY} \to \text{LIMIT}" />
      </TopicSection>

      <TopicSection type="code">
        <h3>ROW_NUMBER — Top-N Per Group</h3>
        <CodeBlock
          language="sql"
          title="row_number.sql"
          code={`-- Table: orders (order_id, user_id, product_id, quantity, unit_price, order_date, status)

-- Get the 3 most recent orders per user
WITH ranked_orders AS (
    SELECT
        user_id,
        order_id,
        order_date,
        quantity * unit_price AS order_total,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY order_date DESC
        ) AS rn
    FROM orders
    WHERE status = 'completed'
)
SELECT user_id, order_id, order_date, order_total
FROM ranked_orders
WHERE rn <= 3
ORDER BY user_id, rn;`}
        />

        <h3>RANK and DENSE_RANK — Leaderboards</h3>
        <CodeBlock
          language="sql"
          title="rank.sql"
          code={`-- Table: sales_reps (rep_id, name, team)
-- Table: deals (deal_id, rep_id, amount, closed_date)

-- Rank reps within each team by total revenue
SELECT
    sr.team,
    sr.name,
    SUM(d.amount)        AS total_revenue,
    RANK()       OVER (PARTITION BY sr.team ORDER BY SUM(d.amount) DESC) AS team_rank,
    DENSE_RANK() OVER (ORDER BY SUM(d.amount) DESC)                     AS global_dense_rank
FROM sales_reps sr
INNER JOIN deals d ON sr.rep_id = d.rep_id
WHERE d.closed_date >= '2025-01-01'
GROUP BY sr.team, sr.name
ORDER BY sr.team, team_rank;`}
        />

        <h3>LAG and LEAD — Row-to-Row Comparisons</h3>
        <CodeBlock
          language="sql"
          title="lag_lead.sql"
          code={`-- Table: daily_metrics (metric_date, product_id, revenue, active_users)

-- Day-over-day and week-over-week revenue change
SELECT
    metric_date,
    product_id,
    revenue,
    LAG(revenue, 1) OVER (
        PARTITION BY product_id ORDER BY metric_date
    ) AS prev_day_revenue,
    revenue - LAG(revenue, 1) OVER (
        PARTITION BY product_id ORDER BY metric_date
    ) AS dod_change,
    ROUND(
        (revenue - LAG(revenue, 1) OVER (
            PARTITION BY product_id ORDER BY metric_date
        )) * 100.0 / NULLIF(LAG(revenue, 1) OVER (
            PARTITION BY product_id ORDER BY metric_date
        ), 0),
        2
    ) AS dod_pct_change,
    LAG(revenue, 7) OVER (
        PARTITION BY product_id ORDER BY metric_date
    ) AS prev_week_revenue
FROM daily_metrics
WHERE metric_date >= '2025-01-01'
ORDER BY product_id, metric_date;`}
        />

        <h3>Running Totals and Moving Averages</h3>
        <CodeBlock
          language="sql"
          title="running_totals.sql"
          code={`-- Table: orders (order_id, user_id, product_id, quantity, unit_price, order_date, status)

-- Cumulative revenue by day, plus 7-day moving average
WITH daily_revenue AS (
    SELECT
        DATE(order_date) AS order_day,
        SUM(quantity * unit_price) AS day_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_date)
)
SELECT
    order_day,
    day_revenue,
    SUM(day_revenue) OVER (
        ORDER BY order_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue,
    AVG(day_revenue) OVER (
        ORDER BY order_day
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d,
    COUNT(*) OVER (
        ORDER BY order_day
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS days_in_window  -- useful to know if you have a full 7-day window
FROM daily_revenue
ORDER BY order_day;`}
        />

        <h3>NTILE — Percentile Buckets</h3>
        <CodeBlock
          language="sql"
          title="ntile.sql"
          code={`-- Table: users (user_id, name, signup_date)
-- Table: orders (order_id, user_id, quantity, unit_price, status)

-- Segment users into spend quartiles
WITH user_spend AS (
    SELECT
        u.user_id,
        u.name,
        SUM(o.quantity * o.unit_price) AS total_spend
    FROM users u
    INNER JOIN orders o ON u.user_id = o.user_id
    WHERE o.status = 'completed'
    GROUP BY u.user_id, u.name
)
SELECT
    user_id,
    name,
    total_spend,
    NTILE(4) OVER (ORDER BY total_spend DESC) AS spend_quartile
    -- 1 = top 25%, 2 = next 25%, etc.
FROM user_spend
ORDER BY spend_quartile, total_spend DESC;`}
        />

        <h3>FIRST_VALUE and LAST_VALUE</h3>
        <CodeBlock
          language="sql"
          title="first_last_value.sql"
          code={`-- Table: events (event_id, user_id, event_type, page_url, created_at)

-- For each user event, show their first and most recent page
SELECT
    user_id,
    created_at,
    page_url,
    FIRST_VALUE(page_url) OVER (
        PARTITION BY user_id
        ORDER BY created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS landing_page,
    LAST_VALUE(page_url) OVER (
        PARTITION BY user_id
        ORDER BY created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS latest_page
FROM events
ORDER BY user_id, created_at;`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>ROW_NUMBER for deduplication</strong>: The most common real-world use. When you have duplicate
            records, assign ROW_NUMBER() OVER (PARTITION BY unique_key ORDER BY updated_at DESC) and keep only rn = 1
            to get the latest version of each record.
          </li>
          <li>
            <strong>Named windows for readability</strong>: If you use the same OVER clause repeatedly, define it
            once with WINDOW w AS (PARTITION BY ... ORDER BY ...) and reference it as OVER w. This reduces
            duplication and errors.
          </li>
          <li>
            <strong>Be explicit about frame clauses</strong>: The default frame with ORDER BY is ROWS BETWEEN
            UNBOUNDED PRECEDING AND CURRENT ROW (running total). Without ORDER BY, it is the entire partition.
            Always specify the frame explicitly to make your intent clear and avoid surprises.
          </li>
          <li>
            <strong>Window functions cannot be nested</strong>: You cannot write LAG(SUM(...) OVER (...), 1)
            OVER (...). Instead, compute the first window function in a CTE, then apply the second in an outer query.
          </li>
          <li>
            <strong>Performance</strong>: Window functions require sorting. If your PARTITION BY and ORDER BY columns
            are indexed, performance is much better. On very large datasets, consider pre-aggregating before applying
            window functions.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Filtering directly on window functions in WHERE</strong>: You cannot write
            WHERE ROW_NUMBER() OVER (...) = 1. Window functions are computed after WHERE. Wrap the query in a CTE
            or subquery and filter in the outer query.
          </li>
          <li>
            <strong>LAST_VALUE with the default frame</strong>: The default frame is ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW, so LAST_VALUE returns the <em>current</em> row, not the actual last row of the partition.
            Fix with ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING.
          </li>
          <li>
            <strong>ROWS vs RANGE</strong>: ROWS counts physical rows. RANGE groups rows with the same ORDER BY
            value together. With duplicate dates, RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW includes all rows
            with the same date as the current row, which can produce unexpected running totals.
          </li>
          <li>
            <strong>Missing PARTITION BY</strong>: Omitting PARTITION BY means the window is the entire result set.
            ROW_NUMBER() OVER (ORDER BY created_at) gives a global row number — which may be what you want, but often
            you intended per-user or per-group numbering.
          </li>
          <li>
            <strong>Non-deterministic ROW_NUMBER</strong>: If two rows have the same ORDER BY value, ROW_NUMBER
            assigns them arbitrarily. Add a tiebreaker column (e.g., ORDER BY created_at DESC, event_id DESC) to
            ensure deterministic results.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Given a table <code>logins(user_id, login_date)</code> (one row per user per day
          they logged in), find the longest streak of consecutive login days for each user.
        </p>
        <CodeBlock
          language="sql"
          title="interview_solution.sql"
          code={`-- Classic "gaps and islands" solved with window functions
-- Step 1: Assign row numbers within each user, ordered by date
-- Step 2: Subtract row_number days from login_date
--         Consecutive dates produce the same "group date"
-- Step 3: Count consecutive days per group, then take the max

WITH login_groups AS (
    SELECT
        user_id,
        login_date,
        login_date - (ROW_NUMBER() OVER (
            PARTITION BY user_id ORDER BY login_date
        ))::int AS streak_group
    FROM logins
),
streak_lengths AS (
    SELECT
        user_id,
        streak_group,
        COUNT(*)         AS streak_length,
        MIN(login_date)  AS streak_start,
        MAX(login_date)  AS streak_end
    FROM login_groups
    GROUP BY user_id, streak_group
)
SELECT
    user_id,
    MAX(streak_length)  AS longest_streak,
    -- Also get the start/end dates of the longest streak
    MAX(streak_length) || ' days ('
        || MIN(CASE WHEN streak_length = (
            SELECT MAX(streak_length) FROM streak_lengths s2
            WHERE s2.user_id = streak_lengths.user_id
        ) THEN streak_start END)::text
        || ' to '
        || MAX(CASE WHEN streak_length = (
            SELECT MAX(streak_length) FROM streak_lengths s2
            WHERE s2.user_id = streak_lengths.user_id
        ) THEN streak_end END)::text
        || ')' AS streak_detail
FROM streak_lengths
GROUP BY user_id
ORDER BY longest_streak DESC;`}
        />
        <p>
          <strong>Key insight:</strong> The trick is that for consecutive dates, <InlineMath math="\text{date} - \text{ROW\_NUMBER()}" /> produces
          a constant value. This converts a &quot;consecutive days&quot; problem into a simple GROUP BY. This is
          the foundational technique behind the gaps-and-islands pattern, which appears in countless data
          engineering and analytics interview questions.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li>
            <strong>&quot;T-SQL Window Functions&quot; by Itzik Ben-Gan</strong> — The most comprehensive book on
            window functions, with deep coverage of frame specifications and optimization.
          </li>
          <li>
            <strong>PostgreSQL Window Functions documentation</strong> — Excellent reference with clear examples
            for every function type and frame option.
          </li>
          <li>
            <strong>Mode Analytics: Window Functions tutorial</strong> — Interactive exercises that build from
            basic ROW_NUMBER to complex running calculations.
          </li>
          <li>
            <strong>Advanced SQL Puzzles by Amit Bansal</strong> — A collection of tricky problems where window
            functions are the key to elegant solutions.
          </li>
        </ul>
      </TopicSection>
    </div>
  );
}
