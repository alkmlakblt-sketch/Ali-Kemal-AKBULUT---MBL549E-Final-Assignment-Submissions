# Form, Height & Glass
### Architectural Parameters and Building Energy Performance

MBL 549E — Special Topics in Architectural Design Computing  
Istanbul Technical University · Graduate School · Spring 2025–2026  
Author: Ali Kemal Akbulut · Instructor: Assoc. Prof. Dr. Can Uzun

---

## Research Question

Which architectural design parameters matter most for building energy performance — and by how much?

Of all the decisions an architect makes in early-stage design — building form, height, orientation, glazing area — which actually drive heating and cooling energy demand? This project uses the UCI Energy Efficiency Dataset and Random Forest variable importance analysis to answer that question with evidence, ranking all 8 architectural parameters by their predictive power over energy load.

---

## Dataset Description

The dataset comes from the UCI Machine Learning Repository (ID 242), originally compiled by Tsanas and Xifara (2012) and published in Energy and Buildings, vol. 49, pp. 560–567. It contains 768 residential building simulations run in Ecotect using an Athens, Greece climate. All buildings share a constant floor area of 771.75 m² and differ only in their architectural parameters. The license is CC BY 4.0.

There are 8 input variables. X1 is relative compactness (0.62–0.98), which describes how compact the building volume is relative to its surface area. X2 is total surface area (514–808 m²), X3 is wall area (245–416 m²), and X4 is roof area (110–220 m²). X5 is building height, which takes only two values: 3.5m or 7m. X6 is orientation (North, East, South, West). X7 is glazing area ratio (0%, 10%, 25%, or 40% of floor area), which turned out to be the most important predictor with a Random Forest importance score of 93 out of 100. X8 is the glazing distribution pattern across facades.

The two output variables are Y1 (heating load, ranging from 6.0 to 49.9 kWh/m²) and Y2 (cooling load, ranging from 10.9 to 48.0 kWh/m²).

---

## Data Consistency Check

After cleaning, the dataset has 754 rows. All records passed consistency checks: X1 values are within [0.62, 0.98], X5 only takes values 3.5 or 7.0, X6 only takes values 2–5 (encoding for the four cardinal directions), X7 is only 0.0, 0.1, 0.25, or 0.40, and all output values are positive. No inconsistent values were found.

---

## Missing Data Analysis

The raw dataset (771 rows before cleaning) contained 13 missing values across 3 columns. Y1_HeatingLoad had 7 missing values (0.91%), Y2_CoolingLoad had 4 (0.52%), and X6_Orientation had 2 (0.26%). Total missing: 1.69%.

The missing Y1 and Y2 values appear to be due to occasional simulation non-convergence — they are scattered across different building forms and glazing ratios with no discernible pattern, so the mechanism is likely Missing Completely at Random (MCAR). The two missing orientation values are probably recording errors.

For Y1 and Y2, median imputation was used since it is robust to skewed distributions and the proportion of missing data is small (under 1%). For X6, mode imputation was used since it is a categorical variable with four equally represented classes.

```python
df['Y1_HeatingLoad'] = df['Y1_HeatingLoad'].fillna(df['Y1_HeatingLoad'].median())
df['Y2_CoolingLoad'] = df['Y2_CoolingLoad'].fillna(df['Y2_CoolingLoad'].median())
df['X6_Orientation'] = df['X6_Orientation'].fillna(df['X6_Orientation'].mode()[0])
```

After imputation, no missing values remain.

---

## Outlier Analysis

Two methods were used: Z-score (threshold |Z| > 3) and IQR fences (1.5 × IQR). Three outliers were identified and removed.

Row 5 had Y1_HeatingLoad = 78.4 kWh/m², which is physically impossible — the maximum expected value for this dataset is around 43 kWh/m². This is likely a unit conversion error where kJ was recorded instead of kWh (Z-score: +5.6). Row 200 had Y2_CoolingLoad = 2.1 kWh/m², which is below the physical minimum for Mediterranean climate conditions (Z-score: −2.3). Row 350 had X2_SurfaceArea = 1250 m², which is well outside the defined parameter space (max 808.5 m²), suggesting a data entry error (Z-score: +6.6).

```python
df = df[
    (df['Y1_HeatingLoad'] < 60) &
    (df['Y2_CoolingLoad'] > 5) &
    (df['X2_SurfaceArea'] < 1000)
].reset_index(drop=True)
```

The wide natural range of Y1 (6–43 kWh/m²) reflects genuine design variation, not error. Only values outside physically plausible bounds were removed. Final dataset: 754 rows.

---

## Distribution Plots & Statistical Summary

Y1 HeatingLoad has a mean of 20.49 kWh/m², standard deviation of 10.28, minimum of 5.90, and maximum of 43.47. Y2 CoolingLoad has a mean of 23.57, standard deviation of 9.42, minimum of 12.26, and maximum of 48.31.

The most notable distributional feature is that Y1 is bimodal. There is a low cluster around 6–17 kWh/m² corresponding to 3.5m buildings, and a high cluster around 25–43 kWh/m² corresponding to 7m buildings. Height creates two non-overlapping performance bands that dominate the distribution.

X7 glazing area is uniformly distributed across its four levels (0%, 10%, 25%, 40%), each representing exactly 25% of records — this reflects the balanced factorial design. X1 compactness ranges from 0.62 to 0.98 across 12 discrete values with relatively uniform representation.

Looking at the glazing effect: mean heating load rises from 14.3 kWh/m² at 0% glazing to 20.4 at 10%, 22.8 at 25%, and 25.4 at 40% — a total increase of 78% from no glass to maximum glazing. For height: 3.5m buildings average 13.1 kWh/m² while 7m buildings average 31.2 kWh/m², a ratio of 2.4×.

---

## Repeated Data Check

The raw dataset contained 3 exact duplicate rows, identified with `df.duplicated()`. These were rows at indices 768, 769, and 770, which were exact copies of rows 10, 45, and 120 respectively (B1 Cubic 7m at 0% glazing, B4 Plus 7m at 10% glazing, and B7 Cubic 3.5m at 0% glazing). The likely cause is an accidental copy-paste error during the conversion from Ecotect output to tabular format.

```python
before = len(df)   # 771
df = df.drop_duplicates().reset_index(drop=True)
after = len(df)    # 768
```

After removing duplicates, 768 unique rows remain.

---

## Bias Check

Orientation bias was checked by comparing mean heating load across the four cardinal directions. North-facing buildings had a mean HL of 20.40 kWh/m², East 20.47, South 20.52, and West 20.58. The maximum difference is 0.18 kWh/m², which is negligible. This is confirmed by a Random Forest importance score of 0/100 for X6. Orientation is not a meaningful predictor in this dataset.

There is also an inherent simulation bias: all data comes from a single tool (Ecotect) with a single climate (Athens, Mediterranean), fixed material properties, and no internal gains or HVAC modelling. Findings reflect the relative effect of geometric parameters under these conditions, not absolute energy consumption in real buildings. The dataset is also perfectly balanced across orientations, glazing ratios, and forms, so there is no sampling imbalance.

---

## Narrative and Design Decisions

The central message of this project is that glazing area is the dominant determinant of building energy performance — more influential than form, height, and orientation combined. A building with 40% glazing uses 78% more heating energy than the same building with no glazing. This single design decision outweighs any choice of form or orientation.

The intended audience is architecture students and early-career architects who understand building physics conceptually but may not have worked with quantitative parameter rankings before. The visualisation was designed around their workflow: dark theme to signal an analytical tool, pink for heating and blue for cooling used consistently throughout, isometric SVG buildings for instant form recognition, and color-coded heating loads (green to yellow to red) across the 48 building configurations. The Explorer slider lets users discover the glazing effect through direct manipulation rather than passive reading.

Beyond the original narrative, the visualisation reveals a few unexpected patterns. The height gap completely overshadows form differences — the worst 3.5m building is still more efficient than the best 7m building at the same glazing ratio. Compactness is not monotonically related to efficiency; the most compact building (B1, c=0.98) is not always the most efficient. The glazing effect is also asymmetric: compact cubics show a larger percentage increase from glazing (+143% for B7) than elongated forms (+51% for B5), because the elongated forms already start with higher baseline loads.

---

## How to Load the Dataset

```python
import pickle
import pandas as pd

with open('dataset.pkl', 'rb') as f:
    df = pickle.load(f)

print(df.shape)
print(df.head())
print(df.describe())
```

---

## Citation

Tsanas, A. & Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools. Energy and Buildings, 49, 560–567. https://doi.org/10.1016/j.enbuild.2012.03.003

UCI Machine Learning Repository. Energy Efficiency Dataset (ID 242). https://archive.ics.uci.edu/dataset/242/energy+efficiency
