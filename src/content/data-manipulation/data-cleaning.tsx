"use client";

import TopicSection from "@/components/TopicSection";
import CodeBlock from "@/components/CodeBlock";
import { InlineMath, BlockMath } from "@/components/Math";

export default function DataCleaning() {
  return (
    <div>
      <TopicSection type="plain-english">
        <p>
          Data cleaning is where most data science projects spend the majority of their time — commonly cited as 60-80% of the work. Raw data
          from databases, APIs, CSVs, and web scrapes is messy: values are missing, types are wrong, duplicates are lurking, outliers are
          distorting statistics, and inconsistent formatting makes analysis unreliable. If you feed garbage into a model, you get garbage out.
          No algorithm can compensate for dirty data.
        </p>
        <p>
          <strong>Missing values</strong> are inevitable in real datasets. A user skips a survey question, a sensor drops a reading, a join
          produces NULLs. The critical question is not just &quot;how do I fill them in&quot; but &quot;WHY are they missing?&quot; Data can be
          missing completely at random (MCAR), missing at random conditional on other observed variables (MAR), or missing not at random (MNAR —
          the hardest case, where the missingness depends on the missing value itself). Your imputation strategy must match the mechanism, or
          you introduce bias.
        </p>
        <p>
          <strong>Outliers</strong> are data points that deviate substantially from the rest. Some outliers are errors (a sensor reading of
          -9999, a human age of 250); these should be removed or corrected. Others are genuine but extreme observations (a billionaire in an
          income dataset); these often need special handling — capping, transforming, or using robust methods that are not sensitive to them.
        </p>
        <p>
          <strong>Type casting</strong> ensures each column has the correct data type. A column of &quot;prices&quot; stored as strings cannot be
          summed. A column of dates stored as integers cannot be used for time-based operations. Getting types right early saves hours of
          debugging later and can reduce memory usage by 10x or more.
        </p>
      </TopicSection>

      <TopicSection type="math">
        <h3>Missing Data Mechanisms</h3>
        <p>
          Let <InlineMath math="X" /> be the complete data and <InlineMath math="M" /> be the missingness indicator
          matrix (<InlineMath math="M_{ij} = 1" /> if <InlineMath math="X_{ij}" /> is missing). Let <InlineMath math="X_{obs}" /> and <InlineMath math="X_{mis}" /> denote
          the observed and missing parts:
        </p>
        <ul>
          <li><strong>MCAR</strong>: <InlineMath math="P(M | X_{obs}, X_{mis}) = P(M)" /> — missingness is independent of all data</li>
          <li><strong>MAR</strong>: <InlineMath math="P(M | X_{obs}, X_{mis}) = P(M | X_{obs})" /> — missingness depends only on observed values</li>
          <li><strong>MNAR</strong>: <InlineMath math="P(M | X_{obs}, X_{mis})" /> depends on <InlineMath math="X_{mis}" /> — missingness depends on the missing values themselves</li>
        </ul>

        <h3>Outlier Detection: Z-Score</h3>
        <p>A data point <InlineMath math="x_i" /> is flagged as an outlier if:</p>
        <BlockMath math="|z_i| = \left|\frac{x_i - \bar{x}}{s}\right| > 3" />
        <p>
          This assumes approximately normal data. For non-normal distributions, use the <strong>modified Z-score</strong> based on the median
          absolute deviation (MAD):
        </p>
        <BlockMath math="\text{MAD} = \text{median}(|x_i - \text{median}(x)|)" />
        <BlockMath math="M_i = \frac{0.6745 \cdot (x_i - \text{median}(x))}{\text{MAD}}" />
        <p>
          The constant 0.6745 makes the MAD consistent with the standard deviation under normality.
        </p>

        <h3>IQR Method</h3>
        <p>The interquartile range method defines outlier bounds:</p>
        <BlockMath math="\text{IQR} = Q_3 - Q_1" />
        <BlockMath math="\text{Lower fence} = Q_1 - 1.5 \cdot \text{IQR}, \quad \text{Upper fence} = Q_3 + 1.5 \cdot \text{IQR}" />
        <p>
          Values outside the fences are outliers. This method is non-parametric — it makes no assumptions about the distribution.
        </p>

        <h3>Mean Imputation Bias</h3>
        <p>
          Replacing missing values with the column mean preserves the mean but <strong>reduces variance</strong>:
        </p>
        <BlockMath math="\text{Var}(X_{\text{imputed}}) = \left(1 - \frac{m}{n}\right) \text{Var}(X_{\text{observed}})" />
        <p>
          where <InlineMath math="m" /> is the number of imputed values and <InlineMath math="n" /> is the total. With 30% missing,
          variance shrinks by 30%. This distorts correlations and makes confidence intervals too narrow.
        </p>
      </TopicSection>

      <TopicSection type="code">
        <h3>Diagnosing Missing Values</h3>
        <CodeBlock
          language="python"
          title="missing_values_diagnosis.py"
          code={`import pandas as pd
import numpy as np

# --- Create realistic messy dataset ---
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'age': np.random.randint(18, 80, n).astype(float),
    'income': np.random.lognormal(10.5, 0.8, n),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], n,
                                   p=[0.3, 0.35, 0.2, 0.1, 0.05]),
    'satisfaction': np.random.choice([1, 2, 3, 4, 5, np.nan], n,
                                      p=[0.05, 0.1, 0.3, 0.3, 0.15, 0.10]),
    'signup_date': pd.date_range('2020-01-01', periods=n, freq='8h'),
})

# Inject realistic missing patterns
# High-income people less likely to report income (MNAR)
mask_income = (df['income'] > 80000) & (np.random.random(n) < 0.3)
df.loc[mask_income, 'income'] = np.nan

# Random missing ages (MCAR)
df.loc[np.random.random(n) < 0.05, 'age'] = np.nan

# --- Diagnosis ---
# 1. Count missing per column
print("Missing counts:")
print(df.isnull().sum())
print(f"\\nMissing percentages:")
print((df.isnull().mean() * 100).round(1))

# 2. Missing patterns: which columns are missing together?
print(f"\\nRows with ANY missing: {df.isnull().any(axis=1).sum()}")
print(f"Rows with ALL complete: {df.notnull().all(axis=1).sum()}")

# 3. Check if missingness correlates with other variables
# (If income missingness correlates with observed variables, it may be MAR/MNAR)
df['income_missing'] = df['income'].isnull().astype(int)
print(f"\\nMean age when income present: {df.loc[~df['income'].isnull(), 'age'].mean():.1f}")
print(f"Mean age when income missing:  {df.loc[df['income'].isnull(), 'age'].mean():.1f}")`}
        />

        <h3>Handling Missing Values</h3>
        <CodeBlock
          language="python"
          title="imputation_strategies.py"
          code={`import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# --- Strategy 1: Drop rows (only if MCAR and < 5% missing) ---
df_dropped = df.dropna(subset=['age', 'income'])
print(f"After dropping: {len(df_dropped)} / {len(df)} rows remain")

# --- Strategy 2: Simple imputation ---
# Mean/median for numeric, mode for categorical
df_imputed = df.copy()
df_imputed['age'] = df_imputed['age'].fillna(df_imputed['age'].median())
df_imputed['income'] = df_imputed['income'].fillna(df_imputed['income'].median())
df_imputed['education'] = df_imputed['education'].fillna(df_imputed['education'].mode()[0])
df_imputed['satisfaction'] = df_imputed['satisfaction'].fillna(df_imputed['satisfaction'].median())

# --- Strategy 3: Group-wise imputation (better for MAR) ---
# Fill income with median income of same education group
df['income_filled'] = df.groupby('education')['income'].transform(
    lambda x: x.fillna(x.median())
)
# Handle groups that are entirely NaN
df['income_filled'] = df['income_filled'].fillna(df['income'].median())

# --- Strategy 4: KNN Imputer (uses similar rows) ---
numeric_cols = ['age', 'income', 'satisfaction']
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = df.copy()
df_knn[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

# --- Strategy 5: Iterative Imputer (MICE-style) ---
# Models each feature as a function of other features
iter_imputer = IterativeImputer(max_iter=10, random_state=42)
df_mice = df.copy()
df_mice[numeric_cols] = iter_imputer.fit_transform(df[numeric_cols])

# --- Strategy 6: Add a missingness indicator flag ---
# Sometimes the FACT that a value is missing is informative
df['income_was_missing'] = df['income'].isnull().astype(int)
# Then impute the value AND keep the flag as a feature

# --- IMPORTANT: Always impute AFTER train/test split ---
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df[numeric_cols], test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),       # fit on TRAIN only
    columns=numeric_cols, index=X_train.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),            # transform TEST with TRAIN stats
    columns=numeric_cols, index=X_test.index
)`}
        />

        <h3>Outlier Detection and Treatment</h3>
        <CodeBlock
          language="python"
          title="outlier_handling.py"
          code={`import pandas as pd
import numpy as np

# Sample data with outliers
np.random.seed(42)
salaries = np.concatenate([
    np.random.normal(70000, 15000, 950),    # Normal salaries
    np.random.normal(500000, 100000, 30),   # Executive salaries
    np.array([-5000, 0, 9999999]),          # Data entry errors
])
df = pd.DataFrame({'salary': salaries})

# --- Method 1: Z-Score ---
mean, std = df['salary'].mean(), df['salary'].std()
df['z_score'] = (df['salary'] - mean) / std
z_outliers = df['z_score'].abs() > 3
print(f"Z-score outliers: {z_outliers.sum()}")

# --- Method 2: Modified Z-Score (robust to outliers themselves) ---
median = df['salary'].median()
mad = np.median(np.abs(df['salary'] - median))
df['modified_z'] = 0.6745 * (df['salary'] - median) / mad
modified_outliers = df['modified_z'].abs() > 3.5
print(f"Modified Z-score outliers: {modified_outliers.sum()}")

# --- Method 3: IQR Method ---
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
iqr_outliers = (df['salary'] < lower_fence) | (df['salary'] > upper_fence)
print(f"IQR outliers: {iqr_outliers.sum()}")
print(f"Fences: [{lower_fence:,.0f}, {upper_fence:,.0f}]")

# --- Treatment 1: Cap / Winsorize ---
df['salary_capped'] = df['salary'].clip(lower=lower_fence, upper=upper_fence)

# --- Treatment 2: Log transform (reduces skew) ---
df['salary_log'] = np.log1p(df['salary'].clip(lower=0))  # log1p handles 0

# --- Treatment 3: Robust scaling (uses median/IQR instead of mean/std) ---
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Centers on median, scales by IQR
df['salary_robust'] = scaler.fit_transform(df[['salary']])

# --- Treatment 4: Separate error outliers from genuine extreme values ---
# Errors: negative salaries, impossibly high values
errors = (df['salary'] < 0) | (df['salary'] > 5_000_000)
print(f"Likely data errors: {errors.sum()}")
df_clean = df[~errors].copy()

# Then handle genuine extremes (executives) with capping
print(f"\\nBefore: mean={df_clean['salary'].mean():,.0f}, median={df_clean['salary'].median():,.0f}")
print(f"After cap: mean={df_clean['salary_capped'].mean():,.0f}, median={df_clean['salary_capped'].median():,.0f}")`}
        />

        <h3>Type Casting and Data Validation</h3>
        <CodeBlock
          language="python"
          title="type_casting.py"
          code={`import pandas as pd
import numpy as np

# --- Simulate messy raw data (e.g., from CSV import) ---
raw = pd.DataFrame({
    'id': ['001', '002', '003', '004', '005'],
    'price': ['29.99', '45.50', 'N/A', '12.00', '$89.99'],
    'quantity': ['10', '5', '3', 'two', '7'],
    'date': ['2024-01-15', '01/20/2024', 'Jan 25, 2024', '2024-02-01', '2024-02-10'],
    'is_premium': ['True', 'False', 'yes', '1', 'no'],
    'category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Clothing'],
    'rating': ['4.5', '3.8', '4.2', '4.7', '4.9'],
    'zipcode': ['10001', '90210', '60614', '77001', '10001'],
})

print("Raw dtypes:")
print(raw.dtypes)  # Everything is 'object' (string)

# --- Clean price: remove $, convert N/A, cast to float ---
df = raw.copy()
df['price'] = (df['price']
    .str.replace('$', '', regex=False)
    .str.replace('N/A', '', regex=False)
    .replace('', np.nan)
    .astype(float)
)

# --- Clean quantity: coerce errors to NaN ---
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')  # 'two' -> NaN

# --- Parse dates with mixed formats ---
df['date'] = pd.to_datetime(df['date'], format='mixed')

# --- Boolean mapping ---
bool_map = {'True': True, 'False': False, 'yes': True, 'no': False, '1': True, '0': False}
df['is_premium'] = df['is_premium'].map(bool_map)

# --- Categorical for low-cardinality strings (saves memory) ---
df['category'] = df['category'].astype('category')
print(f"\\nMemory: object={raw['category'].memory_usage()} bytes, "
      f"category={df['category'].memory_usage()} bytes")

# --- Rating to float ---
df['rating'] = df['rating'].astype(float)

# --- Zipcode should stay as string (leading zeros matter!) ---
# df['zipcode'] is already string — do NOT convert to int

print(f"\\nCleaned dtypes:")
print(df.dtypes)
print(f"\\n{df}")

# --- Validation: assert expected ranges ---
assert df['price'].dropna().between(0, 10000).all(), "Price out of range!"
assert df['quantity'].dropna().between(0, 1000).all(), "Quantity out of range!"
assert df['rating'].between(1, 5).all(), "Rating out of range!"
print("\\nAll validations passed.")`}
        />

        <h3>Deduplication</h3>
        <CodeBlock
          language="python"
          title="deduplication.py"
          code={`import pandas as pd

# --- Exact duplicates ---
df = pd.DataFrame({
    'user_id': [1, 2, 1, 3, 2],
    'event': ['click', 'click', 'click', 'purchase', 'click'],
    'timestamp': ['2024-01-01 10:00', '2024-01-01 10:05', '2024-01-01 10:00',
                  '2024-01-01 10:10', '2024-01-01 10:05'],
})

# Find duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")
print(df[df.duplicated(keep=False)])  # Show ALL rows involved in duplicates

# Drop exact duplicates (keep first occurrence)
df_deduped = df.drop_duplicates()

# --- Duplicates on subset of columns ---
# Keep only the LATEST event per user
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_latest = (df
    .sort_values('timestamp', ascending=False)
    .drop_duplicates(subset=['user_id'], keep='first')
    .sort_values('user_id')
)
print(f"\\nLatest event per user:")
print(df_latest)

# --- Fuzzy deduplication: normalize strings first ---
names = pd.Series(['John Smith', 'john  smith', 'JOHN SMITH', 'Jon Smith'])
normalized = (names
    .str.strip()
    .str.lower()
    .str.replace(r'\\s+', ' ', regex=True)
)
print(f"\\nNormalized names: {normalized.tolist()}")
print(f"Unique after normalization: {normalized.nunique()}")`}
        />
      </TopicSection>

      <TopicSection type="in-practice">
        <ul>
          <li><strong>Always profile your data first</strong>: Before any cleaning, run <code>df.describe()</code>, <code>df.info()</code>, <code>df.isnull().sum()</code>, and check value distributions. Libraries like <code>ydata-profiling</code> generate full reports automatically.</li>
          <li><strong>Impute AFTER train/test split</strong>: Fitting an imputer on the full dataset and then splitting causes data leakage — the test set statistics leak into the training imputer. Always fit on train, transform both.</li>
          <li><strong>Add missingness indicator features</strong>: Before imputing, create binary columns like <code>income_was_missing</code>. The fact that data is missing can itself be a predictive signal (e.g., high-income people declining to report income).</li>
          <li><strong>Domain knowledge trumps statistics</strong>: A statistician might cap outliers at 3 standard deviations. A domain expert knows that a body temperature of 42C is alarming but real, while 420C is a sensor error. Always consult domain context.</li>
          <li><strong>Log your cleaning steps</strong>: Record every transformation (rows dropped, values imputed, outliers capped) so results are reproducible and auditable.</li>
          <li><strong>Validate with assertions</strong>: After cleaning, add assertions: <code>assert df[&apos;age&apos;].between(0, 120).all()</code>. This catches regressions when upstream data changes.</li>
        </ul>
      </TopicSection>

      <TopicSection type="common-mistakes">
        <ul>
          <li><strong>Dropping all rows with any missing value</strong>: <code>df.dropna()</code> can eliminate the majority of your dataset. If you have 20 columns each with 5% missing independently, you lose <InlineMath math="1 - 0.95^{20} \approx 64\%" /> of your rows.</li>
          <li><strong>Mean imputation for skewed distributions</strong>: If income is right-skewed, the mean is pulled high by wealthy outliers. Median imputation is almost always a better default for skewed data.</li>
          <li><strong>Removing outliers blindly</strong>: Automatically removing all outliers can delete real, important data. Executive salaries are outliers but not errors. Always distinguish errors from extreme-but-valid observations.</li>
          <li><strong>Converting zipcodes to integers</strong>: Zipcode &quot;01234&quot; becomes 1234 as an integer, losing the leading zero. Zipcodes, phone numbers, and IDs should stay as strings.</li>
          <li><strong>Imputing the target variable</strong>: Never impute the column you are trying to predict. If the target is missing, drop that row — an imputed target teaches the model nothing real.</li>
          <li><strong>Not checking for duplicates after merging</strong>: Joins can introduce duplicates when keys are not unique. Always verify row counts after every merge.</li>
        </ul>
      </TopicSection>

      <TopicSection type="interview">
        <p><strong>Question:</strong> You receive a dataset with 30% missing values in a key feature. Walk through your decision process for handling this.</p>
        <p><strong>Answer:</strong></p>
        <ol>
          <li><strong>Understand the mechanism</strong>: First determine WHY values are missing. Check if missingness correlates with other columns (e.g., do younger users skip the income field? If so, it&apos;s MAR). Check if the missing values are likely related to the value itself (e.g., high earners declining to report — MNAR). Run <code>df.groupby(df[&apos;feature&apos;].isnull())[other_cols].mean()</code> to compare characteristics of missing vs. non-missing groups.</li>
          <li><strong>If MCAR</strong>: Safe to drop rows (if you can afford the data loss) or use simple imputation (median for numeric, mode for categorical). At 30% missing, dropping is expensive — prefer imputation.</li>
          <li><strong>If MAR</strong>: Use conditional imputation — impute with group medians (e.g., median income per education level) or use KNN/iterative imputer that leverages relationships with other features.</li>
          <li><strong>If MNAR</strong>: This is the hardest case. Create a missingness indicator flag (<code>feature_was_missing = 1</code>), then impute the value. The flag captures the MNAR signal. Consider selection models (Heckman correction) in severe cases.</li>
          <li><strong>Always</strong>: Add the missingness indicator as a separate feature, impute after train/test split, and compare model performance across different imputation strategies. Report the missing rate alongside your results — 30% imputed is materially different from 5% imputed.</li>
        </ol>
      </TopicSection>

      <TopicSection type="go-deeper">
        <ul>
          <li><strong>Flexible Imputation of Missing Data (Stef van Buuren)</strong> — The definitive reference on multiple imputation and the MICE algorithm.</li>
          <li><strong>scikit-learn Imputation Guide</strong> — Practical comparison of SimpleImputer, KNNImputer, and IterativeImputer with code examples.</li>
          <li><strong>Robust Statistics (Huber &amp; Ronchetti)</strong> — Formal treatment of estimation methods that are resistant to outliers.</li>
          <li><strong>Great Expectations library</strong> — Python framework for data validation and quality testing, integrates with pipelines.</li>
        </ul>
      </TopicSection>
    </div>
  );
}
