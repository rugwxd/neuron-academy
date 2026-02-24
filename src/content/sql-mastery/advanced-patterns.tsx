"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function AdvancedPatterns() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Once you master SELECT, JOINs, and window functions, you unlock a class of <strong>advanced SQL
          patterns</strong> that solve real analytical and data engineering problems. These patterns appear
          constantly in data-intensive companies and are the difference between a junior analyst who writes
          one-off queries and a senior engineer who builds reliable, reusable data pipelines.
        </p>
        <p>
          <strong>CTEs (Common Table Expressions)</strong> are the foundation. A CTE is a temporary named result
          set defined with the WITH keyword. It makes complex queries readable by breaking them into logical steps,
          like functions in programming. Recursive CTEs extend this further — they can traverse hierarchies
          (org charts, category trees) and generate series, which is impossible with standard SQL.
        </p>
        <p>
          <strong>Gaps and islands</strong> is a pattern for identifying contiguous sequences in data — consecutive
          login days, uninterrupted subscription periods, or sequential sensor readings. The core technique uses
          the ROW_NUMBER subtraction trick: for consecutive values, subtracting a sequential row number from the
          value produces a constant &quot;island identifier.&quot;
        </p>
        <p>
          <strong>Sessionization</strong> groups a stream of timestamped events into logical sessions. If a user
          clicks at 10:00, 10:02, 10:05, then again at 11:30, the first three clicks form one session and the last
          click starts a new one (using a 30-minute inactivity threshold, for example). This is essential for web
          analytics, product analytics, and behavioral modeling.
          <strong> Funnel analysis</strong> takes this further — given a defined sequence of steps (e.g., visit
          homepage, view product, add to cart, purchase), it measures how many users complete each step and where
          they drop off.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>CTE Execution Model</h3>
        <p>
          A CTE can be thought of as a <strong>named subquery</strong> that is either materialized (computed once,
          stored in temp) or inlined (substituted like a view) depending on the database engine. In PostgreSQL, CTEs
          are optimization fences by default (materialized); in newer versions and other engines, the optimizer can
          inline them.
        </p>

        <h3>Gaps and Islands — The ROW_NUMBER Trick</h3>
        <p>Given an ordered sequence of values <InlineMath math="v_1, v_2, \ldots, v_n" /> and row numbers <InlineMath math="r_1, r_2, \ldots, r_n" />:</p>
        <BlockMath math="\text{island\_id}_i = v_i - r_i" />
        <p>
          For consecutive values, <InlineMath math="v_{i+1} - v_i = 1" /> and <InlineMath math="r_{i+1} - r_i = 1" />,
          so <InlineMath math="\text{island\_id}" /> is constant within each island. A gap in the sequence changes
          the island_id, automatically segmenting the data.
        </p>

        <h3>Sessionization — Inactivity Threshold</h3>
        <p>Given events with timestamps <InlineMath math="t_1 \leq t_2 \leq \ldots \leq t_n" /> and a timeout <InlineMath math="\tau" />:</p>
        <BlockMath math="\text{new\_session}_i = \begin{cases} 1 & \text{if } t_i - t_{i-1} > \tau \text{ or } i = 1 \\ 0 & \text{otherwise} \end{cases}" />
        <BlockMath math="\text{session\_id}_i = \sum_{j=1}^{i} \text{new\_session}_j" />
        <p>
          The session ID is a cumulative sum of the &quot;new session&quot; flag. This translates directly to SQL using LAG
          and a running SUM window function.
        </p>

        <h3>Funnel Conversion Rates</h3>
        <p>For a funnel with steps <InlineMath math="S_1, S_2, \ldots, S_k" />:</p>
        <BlockMath math="\text{conversion\_rate}(S_i \to S_{i+1}) = \frac{|\text{users reaching } S_{i+1}|}{|\text{users reaching } S_i|}" />
        <BlockMath math="\text{overall\_conversion} = \frac{|\text{users reaching } S_k|}{|\text{users reaching } S_1|} = \prod_{i=1}^{k-1} \text{conversion\_rate}(S_i \to S_{i+1})" />
      </TopicSection>

      <TopicSection type="code">
        <h3>CTEs — Building Blocks for Complex Queries</h3>
        <CodeBlock
          language="sql"
          title="ctes_basic.sql"
          code={`-- Table: orders (order_id, user_id, product_id, quantity, unit_price, order_date, status)
-- Table: users (user_id, name, signup_date, country)

-- Multi-step analysis: monthly cohort retention
WITH user_first_order AS (
    -- Step 1: Find each user's first order month
    SELECT
        user_id,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    WHERE status = 'completed'
    GROUP BY user_id
),
user_activity AS (
    -- Step 2: For each user, find all months they were active
    SELECT DISTINCT
        o.user_id,
        ufo.cohort_month,
        DATE_TRUNC('month', o.order_date) AS activity_month
    FROM orders o
    INNER JOIN user_first_order ufo ON o.user_id = ufo.user_id
    WHERE o.status = 'completed'
),
cohort_retention AS (
    -- Step 3: Calculate months since first order and count users
    SELECT
        cohort_month,
        EXTRACT(YEAR FROM age(activity_month, cohort_month)) * 12
            + EXTRACT(MONTH FROM age(activity_month, cohort_month))
            AS months_since_first,
        COUNT(DISTINCT user_id) AS active_users
    FROM user_activity
    GROUP BY cohort_month, activity_month
)
-- Step 4: Final output with retention rate
SELECT
    cohort_month,
    months_since_first,
    active_users,
    ROUND(
        active_users * 100.0 / FIRST_VALUE(active_users) OVER (
            PARTITION BY cohort_month ORDER BY months_since_first
        ),
        1
    ) AS retention_pct
FROM cohort_retention
ORDER BY cohort_month, months_since_first;`}
        />

        <h3>Recursive CTE — Traversing an Org Chart</h3>
        <CodeBlock
          language="sql"
          title="recursive_cte.sql"
          code={`-- Table: employees (emp_id, name, manager_id, department, salary)

-- Build the full reporting chain for every employee
WITH RECURSIVE org_tree AS (
    -- Base case: top-level managers (no manager)
    SELECT
        emp_id,
        name,
        manager_id,
        name::text              AS reporting_chain,
        0                       AS depth
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive step: find direct reports of current level
    SELECT
        e.emp_id,
        e.name,
        e.manager_id,
        ot.reporting_chain || ' > ' || e.name,
        ot.depth + 1
    FROM employees e
    INNER JOIN org_tree ot ON e.manager_id = ot.emp_id
)
SELECT emp_id, name, depth, reporting_chain
FROM org_tree
ORDER BY reporting_chain;

-- Count total reports (direct + indirect) for each manager
WITH RECURSIVE all_reports AS (
    SELECT emp_id, manager_id
    FROM employees
    WHERE manager_id IS NOT NULL

    UNION ALL

    SELECT ar.emp_id, e.manager_id
    FROM all_reports ar
    INNER JOIN employees e ON ar.manager_id = e.emp_id
    WHERE e.manager_id IS NOT NULL
)
SELECT
    m.emp_id,
    m.name,
    COUNT(*) AS total_reports
FROM all_reports ar
INNER JOIN employees m ON ar.manager_id = m.emp_id
GROUP BY m.emp_id, m.name
ORDER BY total_reports DESC;`}
        />

        <h3>Gaps and Islands — Subscription Periods</h3>
        <CodeBlock
          language="sql"
          title="gaps_and_islands.sql"
          code={`-- Table: subscription_status (user_id, status_date, is_active)
-- One row per user per day indicating if subscription is active

-- Find continuous active subscription periods
WITH island_groups AS (
    SELECT
        user_id,
        status_date,
        is_active,
        status_date - (ROW_NUMBER() OVER (
            PARTITION BY user_id, is_active
            ORDER BY status_date
        ))::int AS island_id
    FROM subscription_status
    WHERE is_active = true
)
SELECT
    user_id,
    MIN(status_date) AS subscription_start,
    MAX(status_date) AS subscription_end,
    COUNT(*)         AS days_active
FROM island_groups
GROUP BY user_id, island_id
ORDER BY user_id, subscription_start;

-- Alternative: Find gaps between orders (days with no purchase)
-- Table: orders (order_id, user_id, order_date, status)
WITH user_order_dates AS (
    SELECT DISTINCT
        user_id,
        DATE(order_date) AS order_day
    FROM orders
    WHERE status = 'completed'
),
with_prev AS (
    SELECT
        user_id,
        order_day,
        LAG(order_day) OVER (
            PARTITION BY user_id ORDER BY order_day
        ) AS prev_order_day
    FROM user_order_dates
)
SELECT
    user_id,
    prev_order_day AS gap_start,
    order_day      AS gap_end,
    order_day - prev_order_day AS gap_days
FROM with_prev
WHERE order_day - prev_order_day > 30  -- gaps longer than 30 days
ORDER BY gap_days DESC;`}
        />

        <h3>Sessionization — Grouping Events into Sessions</h3>
        <CodeBlock
          language="sql"
          title="sessionization.sql"
          code={`-- Table: events (event_id, user_id, event_type, page_url, created_at)

-- Sessionize clickstream data with 30-minute inactivity timeout
WITH time_gaps AS (
    SELECT
        user_id,
        event_id,
        event_type,
        page_url,
        created_at,
        EXTRACT(EPOCH FROM (
            created_at - LAG(created_at) OVER (
                PARTITION BY user_id ORDER BY created_at
            )
        )) / 60.0 AS minutes_since_last  -- gap in minutes
    FROM events
),
session_flags AS (
    SELECT
        *,
        CASE
            WHEN minutes_since_last IS NULL     THEN 1  -- first event
            WHEN minutes_since_last > 30        THEN 1  -- new session
            ELSE 0
        END AS is_new_session
    FROM time_gaps
),
sessions AS (
    SELECT
        *,
        SUM(is_new_session) OVER (
            PARTITION BY user_id ORDER BY created_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_number
    FROM session_flags
)
-- Session-level summary
SELECT
    user_id,
    session_number,
    MIN(created_at)     AS session_start,
    MAX(created_at)     AS session_end,
    COUNT(*)            AS events_in_session,
    EXTRACT(EPOCH FROM (
        MAX(created_at) - MIN(created_at)
    )) / 60.0           AS session_duration_minutes,
    ARRAY_AGG(page_url ORDER BY created_at) AS page_sequence
FROM sessions
GROUP BY user_id, session_number
ORDER BY user_id, session_number;`}
        />

        <h3>Funnel Analysis</h3>
        <CodeBlock
          language="sql"
          title="funnel_analysis.sql"
          code={`-- Table: events (event_id, user_id, event_type, page_url, created_at)
-- event_type values: 'page_view', 'product_view', 'add_to_cart', 'checkout', 'purchase'

-- Strict-order funnel: users must complete steps in sequence
WITH funnel_steps AS (
    SELECT
        user_id,
        MIN(CASE WHEN event_type = 'page_view'    THEN created_at END) AS step1_time,
        MIN(CASE WHEN event_type = 'product_view'  THEN created_at END) AS step2_time,
        MIN(CASE WHEN event_type = 'add_to_cart'   THEN created_at END) AS step3_time,
        MIN(CASE WHEN event_type = 'checkout'      THEN created_at END) AS step4_time,
        MIN(CASE WHEN event_type = 'purchase'      THEN created_at END) AS step5_time
    FROM events
    WHERE created_at >= '2025-01-01'
      AND created_at <  '2025-02-01'
    GROUP BY user_id
),
ordered_funnel AS (
    -- Enforce strict ordering: each step must happen after the previous
    SELECT
        user_id,
        step1_time,
        CASE WHEN step2_time > step1_time THEN step2_time END AS step2_time,
        CASE WHEN step3_time > step2_time
             AND  step2_time > step1_time THEN step3_time END AS step3_time,
        CASE WHEN step4_time > step3_time
             AND  step3_time > step2_time
             AND  step2_time > step1_time THEN step4_time END AS step4_time,
        CASE WHEN step5_time > step4_time
             AND  step4_time > step3_time
             AND  step3_time > step2_time
             AND  step2_time > step1_time THEN step5_time END AS step5_time
    FROM funnel_steps
)
SELECT
    'page_view'    AS step,
    1 AS step_num,
    COUNT(step1_time) AS users_reached,
    100.0 AS pct_of_total
FROM ordered_funnel
UNION ALL
SELECT 'product_view', 2,
    COUNT(step2_time),
    ROUND(COUNT(step2_time) * 100.0 / NULLIF(COUNT(step1_time), 0), 1)
FROM ordered_funnel
UNION ALL
SELECT 'add_to_cart', 3,
    COUNT(step3_time),
    ROUND(COUNT(step3_time) * 100.0 / NULLIF(COUNT(step1_time), 0), 1)
FROM ordered_funnel
UNION ALL
SELECT 'checkout', 4,
    COUNT(step4_time),
    ROUND(COUNT(step4_time) * 100.0 / NULLIF(COUNT(step1_time), 0), 1)
FROM ordered_funnel
UNION ALL
SELECT 'purchase', 5,
    COUNT(step5_time),
    ROUND(COUNT(step5_time) * 100.0 / NULLIF(COUNT(step1_time), 0), 1)
FROM ordered_funnel
ORDER BY step_num;`}
        />

        <h3>Pivot with CASE — Crosstab Reports</h3>
        <CodeBlock
          language="sql"
          title="pivot.sql"
          code={`-- Table: orders (order_id, user_id, product_id, quantity, unit_price, order_date, status)
-- Table: products (product_id, product_name, category)

-- Revenue by category per month (pivot)
SELECT
    DATE_TRUNC('month', o.order_date)::date AS month,
    SUM(CASE WHEN p.category = 'electronics' THEN o.quantity * o.unit_price ELSE 0 END) AS electronics,
    SUM(CASE WHEN p.category = 'clothing'    THEN o.quantity * o.unit_price ELSE 0 END) AS clothing,
    SUM(CASE WHEN p.category = 'groceries'   THEN o.quantity * o.unit_price ELSE 0 END) AS groceries,
    SUM(CASE WHEN p.category = 'books'       THEN o.quantity * o.unit_price ELSE 0 END) AS books,
    SUM(o.quantity * o.unit_price) AS total_revenue
FROM orders o
INNER JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY DATE_TRUNC('month', o.order_date)
ORDER BY month;`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li>
            <strong>CTEs are your best friend for readability</strong>: Break every complex query into named steps.
            Future you (and your teammates) will thank you. Name CTEs descriptively — user_first_order is better
            than cte1.
          </li>
          <li>
            <strong>Watch out for recursive CTE infinite loops</strong>: Always ensure the recursive step converges.
            Add a depth counter and a WHERE depth &lt; 100 safety limit. Most databases also have a max recursion
            setting you can configure.
          </li>
          <li>
            <strong>Sessionization thresholds vary by product</strong>: 30 minutes is the Google Analytics default,
            but mobile apps might use 5 minutes, while B2B SaaS might use 60 minutes. Choose the threshold based on
            your product&apos;s usage patterns and validate by looking at the distribution of inter-event gaps.
          </li>
          <li>
            <strong>Funnel analysis requires defining &quot;strict&quot; vs &quot;loose&quot; ordering</strong>: A strict
            funnel requires step 2 to happen after step 1. A loose funnel only checks that both events occurred,
            regardless of order. Strict funnels give more accurate conversion rates but lose users who browse
            non-linearly.
          </li>
          <li>
            <strong>Materialized CTEs in PostgreSQL</strong>: Before v12, all CTEs were materialized (computed once,
            stored in temp). In v12+, simple CTEs may be inlined. Use NOT MATERIALIZED or MATERIALIZED hints to control
            this behavior when performance matters.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li>
            <strong>Using recursive CTEs for problems that do not need recursion</strong>: Generating a date series
            or number sequence is a common use case, but many databases offer built-in functions
            (generate_series in PostgreSQL, CONNECT BY in Oracle) that are faster.
          </li>
          <li>
            <strong>Gaps-and-islands with non-unique dates</strong>: The ROW_NUMBER subtraction trick assumes
            one row per date per group. If a user can have multiple events on the same date, first DISTINCT
            the dates before applying the pattern.
          </li>
          <li>
            <strong>Sessionization without ordering guarantees</strong>: If two events have the exact same timestamp,
            LAG behavior is non-deterministic. Add a tiebreaker column (event_id) to your ORDER BY clause.
          </li>
          <li>
            <strong>Funnel analysis that double-counts users</strong>: If a user views a product page twice,
            then adds to cart once, a naive funnel counts them once at product_view (correct) but might
            count them at add_to_cart only if the timing logic is not right. Use MIN(timestamp) per step per user.
          </li>
          <li>
            <strong>Overly nested subqueries instead of CTEs</strong>: Some analysts write 5-level nested
            subqueries that are impossible to debug. Refactor into CTEs — the performance is identical in most
            modern databases, but readability is dramatically better.
          </li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p>
          <strong>Question:</strong> Given a table <code>user_events(user_id, event_type, event_time)</code> where
          event_type is one of &apos;signup&apos;, &apos;activation&apos;, &apos;first_purchase&apos;, &apos;second_purchase&apos;, write a query that:
          (1) builds a conversion funnel showing how many users reach each step in order, (2) calculates the
          step-over-step and overall conversion rates, and (3) computes the median time between signup and
          first_purchase for users who completed that step.
        </p>
        <CodeBlock
          language="sql"
          title="interview_solution.sql"
          code={`-- Part 1 & 2: Conversion funnel with step-over-step rates
WITH user_steps AS (
    SELECT
        user_id,
        MIN(CASE WHEN event_type = 'signup'          THEN event_time END) AS t_signup,
        MIN(CASE WHEN event_type = 'activation'      THEN event_time END) AS t_activation,
        MIN(CASE WHEN event_type = 'first_purchase'  THEN event_time END) AS t_first_purchase,
        MIN(CASE WHEN event_type = 'second_purchase' THEN event_time END) AS t_second_purchase
    FROM user_events
    GROUP BY user_id
),
ordered_steps AS (
    SELECT
        user_id,
        t_signup,
        CASE WHEN t_activation > t_signup
             THEN t_activation END               AS t_activation,
        CASE WHEN t_first_purchase > t_activation
             AND  t_activation > t_signup
             THEN t_first_purchase END            AS t_first_purchase,
        CASE WHEN t_second_purchase > t_first_purchase
             AND  t_first_purchase > t_activation
             AND  t_activation > t_signup
             THEN t_second_purchase END           AS t_second_purchase
    FROM user_steps
    WHERE t_signup IS NOT NULL
),
funnel_counts AS (
    SELECT
        COUNT(t_signup)          AS signups,
        COUNT(t_activation)      AS activations,
        COUNT(t_first_purchase)  AS first_purchases,
        COUNT(t_second_purchase) AS second_purchases
    FROM ordered_steps
)
SELECT
    step,
    users_reached,
    ROUND(users_reached * 100.0 / NULLIF(prev_step_users, 0), 1) AS step_conversion_pct,
    ROUND(users_reached * 100.0 / NULLIF(total_signups, 0), 1)   AS overall_conversion_pct
FROM (
    SELECT 'signup' AS step, 1 AS step_num, signups AS users_reached,
           signups AS prev_step_users, signups AS total_signups FROM funnel_counts
    UNION ALL
    SELECT 'activation', 2, activations, signups, signups FROM funnel_counts
    UNION ALL
    SELECT 'first_purchase', 3, first_purchases, activations, signups FROM funnel_counts
    UNION ALL
    SELECT 'second_purchase', 4, second_purchases, first_purchases, signups FROM funnel_counts
) sub
ORDER BY step_num;

-- Part 3: Median time from signup to first_purchase
WITH user_steps AS (
    SELECT
        user_id,
        MIN(CASE WHEN event_type = 'signup'         THEN event_time END) AS t_signup,
        MIN(CASE WHEN event_type = 'first_purchase'  THEN event_time END) AS t_first_purchase
    FROM user_events
    GROUP BY user_id
),
time_to_purchase AS (
    SELECT
        user_id,
        EXTRACT(EPOCH FROM (t_first_purchase - t_signup)) / 3600.0 AS hours_to_purchase
    FROM user_steps
    WHERE t_first_purchase > t_signup
)
SELECT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hours_to_purchase) AS median_hours_to_purchase,
    ROUND(AVG(hours_to_purchase), 1) AS mean_hours_to_purchase,
    COUNT(*) AS users_with_purchase
FROM time_to_purchase;`}
        />
        <p>
          <strong>Key insight:</strong> This question tests three skills at once: (1) the CASE-WHEN pivot pattern for
          funnels, (2) enforcing strict ordering of events, and (3) using PERCENTILE_CONT for median calculation.
          The strict ordering is the hardest part — many candidates forget that events can occur out of the expected
          order and produce inflated conversion rates.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li>
            <strong>&quot;SQL for Data Scientists&quot; by Renee Teate</strong> — Covers CTEs, window functions, and
            analytical patterns with real-world datasets.
          </li>
          <li>
            <strong>Ergest Xheblati&apos;s &quot;Gaps and Islands&quot; series</strong> — Deep dive into every variation of the
            gaps-and-islands problem with PostgreSQL examples.
          </li>
          <li>
            <strong>dbt (data build tool) documentation on CTEs</strong> — Best practices for organizing SQL
            transformations into modular, testable CTEs in production data pipelines.
          </li>
          <li>
            <strong>Google BigQuery &quot;Sessionization&quot; cookbook</strong> — Production-grade sessionization patterns
            optimized for large-scale event data.
          </li>
          <li>
            <strong>LeetCode Hard SQL problems</strong> — Problems #601 (Human Traffic of Stadium), #185
            (Department Top Three Salaries), and #262 (Trips and Users) test advanced pattern recognition.
          </li>
        </ul>
      </TopicSection>
    </div>
  );
}
