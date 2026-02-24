"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function Pandas() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Pandas is the workhorse library for structured data in Python. Its central object is the <strong>DataFrame</strong> — a two-dimensional,
          labeled data structure with columns of potentially different types. Think of it as a programmable spreadsheet or an in-memory SQL table.
          Each column is a <strong>Series</strong> (a labeled 1D array), and the DataFrame index provides fast row lookups.
        </p>
        <p>
          <strong>GroupBy</strong> implements the split-apply-combine paradigm: you split data into groups based on one or more keys, apply a
          function (aggregation, transformation, or filter) to each group, and combine the results. This is equivalent to SQL&apos;s
          <code>GROUP BY</code> but far more flexible — you can apply arbitrary Python functions, including custom aggregations that would be
          impossible in SQL.
        </p>
        <p>
          <strong>Merge and Join</strong> operations combine DataFrames horizontally, just like SQL JOINs. Pandas supports inner, left, right,
          outer, and cross joins on one or more keys. Understanding the difference between <code>merge</code>, <code>join</code>, and
          <code>concat</code> is critical — they solve different problems and have different performance characteristics.
        </p>
        <p>
          <strong>Window functions</strong> (rolling, expanding, ewm) let you compute statistics over a sliding window of rows — moving averages,
          cumulative sums, exponentially weighted means. These are essential for time series analysis, feature engineering, and any situation
          where context from neighboring rows matters.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>GroupBy Aggregations</h3>
        <p>
          A groupby-mean computes, for each group <InlineMath math="g" />:
        </p>
        <BlockMath math="\bar{x}_g = \frac{1}{|G_g|} \sum_{i \in G_g} x_i" />
        <p>
          where <InlineMath math="G_g" /> is the set of row indices belonging to group <InlineMath math="g" />.
        </p>

        <h3>Rolling Window Statistics</h3>
        <p>
          A rolling mean with window size <InlineMath math="w" /> at position <InlineMath math="t" />:
        </p>
        <BlockMath math="\text{rolling\_mean}_t = \frac{1}{w} \sum_{i=t-w+1}^{t} x_i" />

        <h3>Exponentially Weighted Moving Average (EWMA)</h3>
        <p>
          With decay parameter <InlineMath math="\alpha \in (0, 1]" /> (where <InlineMath math="\alpha = \frac{2}{\text{span}+1}" />):
        </p>
        <BlockMath math="y_t = \alpha \cdot x_t + (1 - \alpha) \cdot y_{t-1}" />
        <p>
          Recent observations get exponentially more weight. This is widely used in finance for smoothing price series
          and in ML for tracking running statistics during training.
        </p>

        <h3>Join Complexity</h3>
        <p>
          Pandas merge uses a hash-join by default. For two DataFrames of size <InlineMath math="m" /> and <InlineMath math="n" />:
        </p>
        <ul>
          <li>Build phase: <InlineMath math="O(m)" /> to hash the smaller table</li>
          <li>Probe phase: <InlineMath math="O(n)" /> to look up matches</li>
          <li>Total: <InlineMath math="O(m + n)" /> average case, <InlineMath math="O(m \cdot n)" /> worst case (many duplicates)</li>
        </ul>
      </TopicSection>

      <TopicSection type="code">
        <h3>DataFrame Fundamentals</h3>
        <CodeBlock
          language="python"
          title="pandas_basics.py"
          code={`import pandas as pd
import numpy as np

# --- Creating DataFrames ---
df = pd.DataFrame({
    'name':       ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'department': ['Engineering', 'Marketing', 'Engineering', 'Marketing', 'Engineering'],
    'salary':     [120000, 85000, 110000, 90000, 130000],
    'years_exp':  [5, 3, 4, 6, 8],
    'rating':     [4.5, 3.8, 4.2, 4.7, 4.9]
})

# --- Selection ---
# Single column (returns Series)
salaries = df['salary']

# Multiple columns (returns DataFrame)
subset = df[['name', 'salary']]

# .loc — label-based (inclusive on both ends)
df.loc[0:2, 'name':'salary']

# .iloc — integer-based (exclusive end)
df.iloc[0:3, 0:3]

# --- Boolean filtering ---
senior_engineers = df[(df['department'] == 'Engineering') & (df['years_exp'] >= 5)]
print(senior_engineers)

# --- Adding computed columns ---
df['salary_per_year'] = df['salary'] / df['years_exp']
df['is_senior'] = df['years_exp'] >= 5

# --- Sorting ---
df_sorted = df.sort_values('salary', ascending=False)

# --- Quick stats ---
print(df.describe())
print(df.dtypes)
print(df.info())`}
        />

        <h3>GroupBy: Split-Apply-Combine</h3>
        <CodeBlock
          language="python"
          title="groupby_operations.py"
          code={`import pandas as pd
import numpy as np

# Sample dataset: sales transactions
np.random.seed(42)
n = 10000
sales = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=n, freq='h'),
    'store': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], n),
    'revenue': np.random.lognormal(4, 1, n).round(2),
    'units': np.random.randint(1, 20, n),
})

# --- Basic aggregation ---
store_stats = sales.groupby('store').agg(
    total_revenue=('revenue', 'sum'),
    avg_revenue=('revenue', 'mean'),
    total_units=('units', 'sum'),
    num_transactions=('revenue', 'count'),
).round(2)
print(store_stats)

# --- Multiple groupby keys ---
category_by_store = sales.groupby(['store', 'category'])['revenue'].agg(['mean', 'sum', 'count'])
print(category_by_store.head(8))

# --- Transform: broadcast group result back to original shape ---
# Add column: each row's revenue as % of its store's total
sales['pct_of_store'] = (
    sales.groupby('store')['revenue']
    .transform(lambda x: x / x.sum() * 100)
)

# --- Custom aggregation ---
def revenue_iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

store_iqr = sales.groupby('store')['revenue'].agg([revenue_iqr, 'median'])
print(store_iqr)

# --- Filter: keep only groups meeting a condition ---
# Keep only stores with > 2500 transactions
large_stores = sales.groupby('store').filter(lambda g: len(g) > 2500)
print(f"Rows before: {len(sales)}, after: {len(large_stores)}")`}
        />

        <h3>Merge and Join</h3>
        <CodeBlock
          language="python"
          title="merge_join.py"
          code={`import pandas as pd

# --- Setup: two related tables ---
orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4, 5],
    'customer_id': [101, 102, 101, 103, 104],
    'amount': [250, 150, 300, 450, 200],
})

customers = pd.DataFrame({
    'customer_id': [101, 102, 103, 105],
    'name': ['Alice', 'Bob', 'Charlie', 'Eve'],
    'city': ['NYC', 'LA', 'Chicago', 'Houston'],
})

# --- INNER join: only matching rows ---
inner = pd.merge(orders, customers, on='customer_id', how='inner')
print(f"Inner: {len(inner)} rows")  # 4 rows (order 5 dropped: customer 104 not in customers)

# --- LEFT join: keep all from left table ---
left = pd.merge(orders, customers, on='customer_id', how='left')
print(f"Left: {len(left)} rows")    # 5 rows (order 5 has NaN for name/city)

# --- RIGHT join: keep all from right table ---
right = pd.merge(orders, customers, on='customer_id', how='right')
print(f"Right: {len(right)} rows")  # 5 rows (Eve has NaN for order_id/amount)

# --- OUTER join: keep everything ---
outer = pd.merge(orders, customers, on='customer_id', how='outer')
print(f"Outer: {len(outer)} rows")  # 6 rows

# --- Multiple join keys ---
df1 = pd.DataFrame({'year': [2023, 2023, 2024], 'quarter': [1, 2, 1], 'revenue': [100, 200, 150]})
df2 = pd.DataFrame({'year': [2023, 2024, 2024], 'quarter': [2, 1, 2], 'headcount': [50, 55, 60]})
merged = pd.merge(df1, df2, on=['year', 'quarter'], how='inner')
print(merged)

# --- merge_asof: fuzzy time-based join ---
# Join each trade to the most recent quote BEFORE it
trades = pd.DataFrame({
    'time': pd.to_datetime(['10:00:01', '10:00:03', '10:00:05']),
    'ticker': ['AAPL', 'AAPL', 'AAPL'],
    'quantity': [100, 200, 150],
})
quotes = pd.DataFrame({
    'time': pd.to_datetime(['10:00:00', '10:00:02', '10:00:04']),
    'ticker': ['AAPL', 'AAPL', 'AAPL'],
    'bid': [149.5, 149.8, 150.0],
    'ask': [150.0, 150.3, 150.5],
})
result = pd.merge_asof(trades.sort_values('time'), quotes.sort_values('time'),
                        on='time', by='ticker')
print(result)`}
        />

        <h3>Window Functions</h3>
        <CodeBlock
          language="python"
          title="window_functions.py"
          code={`import pandas as pd
import numpy as np

# --- Time series data ---
dates = pd.date_range('2023-01-01', periods=252, freq='B')  # Business days
np.random.seed(42)
price = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.02))  # Geometric Brownian motion

stock = pd.DataFrame({'date': dates, 'price': price})
stock = stock.set_index('date')

# --- Rolling window ---
stock['ma_20'] = stock['price'].rolling(window=20).mean()       # 20-day moving average
stock['ma_50'] = stock['price'].rolling(window=50).mean()       # 50-day moving average
stock['volatility_20'] = stock['price'].rolling(20).std()       # 20-day rolling volatility

# --- Exponentially Weighted Moving Average ---
stock['ewma_20'] = stock['price'].ewm(span=20).mean()

# --- Expanding window (cumulative) ---
stock['cummax'] = stock['price'].expanding().max()              # Running maximum
stock['drawdown'] = stock['price'] / stock['cummax'] - 1       # Drawdown from peak

# --- shift / lag ---
stock['prev_price'] = stock['price'].shift(1)                  # Previous day price
stock['daily_return'] = stock['price'].pct_change()            # Daily % return

# --- Rolling apply with custom function ---
def sharpe_ratio(returns, risk_free_rate=0.0):
    """Annualized Sharpe ratio."""
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / excess.std()

stock['rolling_sharpe'] = (
    stock['daily_return']
    .rolling(60)
    .apply(sharpe_ratio, raw=True)
)

# --- Rank within a rolling window ---
stock['percentile_rank'] = stock['price'].rolling(60).rank(pct=True)

print(stock.dropna().tail(10).round(2))`}
        />

        <h3>Method Chaining</h3>
        <CodeBlock
          language="python"
          title="method_chaining.py"
          code={`import pandas as pd
import numpy as np

# Clean, readable pipelines using method chaining
result = (
    pd.read_csv('sales.csv')
    .rename(columns=str.lower)
    .assign(
        date=lambda df: pd.to_datetime(df['date']),
        revenue=lambda df: df['price'] * df['quantity'],
        month=lambda df: df['date'].dt.to_period('M'),
    )
    .query('revenue > 0 and category != "Returns"')
    .groupby(['month', 'category'], as_index=False)
    .agg(
        total_revenue=('revenue', 'sum'),
        avg_order_value=('revenue', 'mean'),
        order_count=('revenue', 'count'),
    )
    .sort_values(['month', 'total_revenue'], ascending=[True, False])
    .reset_index(drop=True)
)

# .pipe() for custom functions in the chain
def add_yoy_growth(df, value_col):
    """Add year-over-year growth rate."""
    df = df.copy()
    df['yoy_growth'] = df.groupby('category')[value_col].pct_change(12)
    return df

result = result.pipe(add_yoy_growth, 'total_revenue')`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Use <code>.query()</code> for readable filters</strong>: <code>df.query(&apos;age &gt; 25 and city == &quot;NYC&quot;&apos;)</code> is clearer than chaining boolean masks with <code>&amp;</code>.</li>
          <li><strong>Prefer <code>.agg()</code> with named aggregations</strong>: <code>.agg(total=(&apos;revenue&apos;, &apos;sum&apos;))</code> gives you named columns directly, avoiding ambiguous multi-level column names.</li>
          <li><strong>Use categorical dtype for low-cardinality strings</strong>: <code>df[&apos;state&apos;] = df[&apos;state&apos;].astype(&apos;category&apos;)</code> reduces memory 10-50x and speeds up groupby.</li>
          <li><strong>Set proper index</strong>: If you repeatedly filter by a column, set it as the index for O(1) lookups: <code>df.set_index(&apos;user_id&apos;)</code>.</li>
          <li><strong>Use <code>merge_asof</code> for time-based joins</strong>: When timestamps don&apos;t align exactly (trades vs quotes, events vs snapshots), <code>merge_asof</code> joins to the nearest prior match.</li>
          <li><strong>Avoid iterrows()</strong>: It is extremely slow. Use <code>.apply()</code>, vectorized operations, or <code>.itertuples()</code> (10x faster than iterrows) if you must iterate.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>SettingWithCopyWarning</strong>: <code>df[df[&apos;x&apos;] &gt; 0][&apos;y&apos;] = 1</code> does NOT modify <code>df</code> — it modifies a temporary copy. Use <code>df.loc[df[&apos;x&apos;] &gt; 0, &apos;y&apos;] = 1</code> instead.</li>
          <li><strong>Merge explosion</strong>: If both DataFrames have duplicate keys, merge produces a Cartesian product for those keys. A merge of 1000 rows on 1000 rows can produce millions. Always check <code>len(result)</code> after merging.</li>
          <li><strong>Ignoring NaN in groupby</strong>: By default, <code>groupby</code> drops NaN keys. Use <code>groupby(col, dropna=False)</code> to include them.</li>
          <li><strong>Confusing <code>axis=0</code> and <code>axis=1</code></strong>: <code>axis=0</code> means &quot;operate along rows&quot; (i.e., collapse rows → one result per column). <code>axis=1</code> means &quot;operate along columns&quot; (collapse columns → one result per row).</li>
          <li><strong>Using <code>apply</code> when vectorized operations exist</strong>: <code>df[&apos;upper&apos;] = df[&apos;name&apos;].apply(str.upper)</code> is 10x slower than <code>df[&apos;name&apos;].str.upper()</code>.</li>
          <li><strong>Not specifying <code>on=</code> in merge</strong>: If you omit the <code>on</code> parameter, Pandas merges on ALL shared column names, which can produce unexpected results.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> Given a DataFrame of user login events with columns <code>[user_id, login_time]</code>, compute the number of users who logged in on at least 3 consecutive days. No loops allowed.</p>
        <p><strong>Answer:</strong></p>
        <CodeBlock
          language="python"
          title="consecutive_logins.py"
          code={`import pandas as pd
import numpy as np

# Sample data
logins = pd.DataFrame({
    'user_id': [1,1,1,1,1, 2,2,2, 3,3,3,3],
    'login_time': pd.to_datetime([
        '2024-01-01','2024-01-02','2024-01-03','2024-01-05','2024-01-06',
        '2024-01-01','2024-01-03','2024-01-04',
        '2024-01-10','2024-01-11','2024-01-12','2024-01-13',
    ])
})

# Step 1: Get unique login dates per user
daily = (logins
    .assign(login_date=logins['login_time'].dt.date)
    .drop_duplicates(subset=['user_id', 'login_date'])
    .sort_values(['user_id', 'login_date'])
)

# Step 2: Assign a row number within each user
daily['rn'] = daily.groupby('user_id').cumcount()

# Step 3: Subtract row number of days from login_date
# Consecutive dates will produce the SAME "group_key"
daily['login_date'] = pd.to_datetime(daily['login_date'])
daily['group_key'] = daily['login_date'] - pd.to_timedelta(daily['rn'], unit='D')

# Step 4: Count consecutive streak lengths
streaks = (daily
    .groupby(['user_id', 'group_key'])
    .size()
    .reset_index(name='streak_length')
)

# Step 5: Filter users with any streak >= 3
users_with_3_consecutive = streaks.loc[
    streaks['streak_length'] >= 3, 'user_id'
].nunique()

print(f"Users with 3+ consecutive login days: {users_with_3_consecutive}")
# Output: 2 (user 1 has Jan 1-3, user 3 has Jan 10-13)`}
        />
        <p>
          <strong>Key insight</strong>: The &quot;row number subtraction&quot; trick is a classic gaps-and-islands technique. If dates are consecutive,
          subtracting an incrementing integer produces the same constant — this groups consecutive runs together without any loops.
        </p>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Pandas documentation — GroupBy: split-apply-combine</strong> — Comprehensive guide with dozens of examples.</li>
          <li><strong>Modern Pandas (Tom Augspurger)</strong> — Blog series on idiomatic Pandas: method chaining, piping, and performance.</li>
          <li><strong>Effective Pandas (Matt Harrison)</strong> — Book focused on writing clean, performant Pandas code.</li>
          <li><strong>Apache Arrow and Pandas 2.0</strong> — The Arrow backend in Pandas 2.0 brings significant memory and speed improvements, especially for string columns.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
