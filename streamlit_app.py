
# ---------------- Step 00 - Imports ----------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- Page Config ----------------
st.markdown("""
<style>
/* 让所有 border container 看起来像卡片 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    background: rgba(255,255,255,0.04) !important;
    padding: 12px 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Data ----------------
df = pd.read_csv("Student_data.csv")

# ---------------- Sidebar (Default) ----------------
st.sidebar.title("Page Navigation")
page = st.sidebar.selectbox(
    "",
    [
        "Home",
        "Dataset Overview",
        "Visualization & Insights",
        "Regression & Prediction",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 · University Student Performance & Habits")
# =========================================================
# 🏠 HOME PAGE
# =========================================================
if page == "Home":
    st.image("overallpageicon.png", width=30) 
    st.title("University Student Performance & Habits")

    st.markdown("**Group Members:** Coco Dai · Nadalia Jin · Solomon Kim · Aria Zhang")
    st.write("")

    # ---------- TRIANGLE IMAGE (TOP, NOT IN CARD) ----------
    st.image("tri.jpg")
    st.caption("The 3 S's of the S-Triangle that guide a college student's life are: Sleep, Studying, and a Social Life")

    st.write("")

    # ---------- PROJECT OVERVIEW ----------
    with st.container(border=True):
        st.subheader("Project Overview")

        st.markdown(
            """
            This project investigates how students balance the **3-S Triangle**:

            - **Study Time**
            - **Sleep**
            - **Social Life**

            ### Objective
            Identify an *optimal equilibrium point* where students can:

            - Maintain a strong GPA  
            - Preserve healthy sleep habits  
            - Sustain an active social life  
            """
        )

    st.write("")

    # ---------- GOOD GPA SECTION ----------
    with st.container(border=True):
        st.subheader("What is a “Good GPA”? (Top 25% Students)")

        if "Final_CGPA" in df.columns:
            top_25_threshold = df["Final_CGPA"].quantile(0.75)
            top_students = df[df["Final_CGPA"] >= top_25_threshold]
            avg_top_25_gpa = top_students["Final_CGPA"].mean()

            m1, m2 = st.columns(2)
            m1.metric("Top 25% Avg CGPA", f"{avg_top_25_gpa:.2f}")
            m2.metric("Top Students Count", f"{len(top_students)}")

# =========================================================
# 📂 DATASET OVERVIEW
# =========================================================
elif page == "Dataset Overview":

    st.subheader("01 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Cells", int(df.isnull().sum().sum()))

    st.markdown("---")

    st.subheader("Data Preview")
    rows = st.slider("Rows to display", 5, 30, 5)
    st.dataframe(df.head(rows))

    st.subheader("Column Names")
    st.write(list(df.columns))

    st.subheader("Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

    st.markdown("""
    🔗 **Source:**  
    https://www.kaggle.com/datasets/robiulhasanjisan/university-student-performance-and-habits-dataset
    """)

# =========================================================
# 📊 VISUALIZATION & INSIGHTS
# =========================================================
elif page == "Visualization & Insights":

    ## Step 03 - Data Viz
    st.subheader("02 Data Visualization")

    tab1, tab2, tab3, tab4, tab5 =st.tabs(["Pie Charts","Boxplot","Histogram","Scatterplot","Correlation Heatmap"])

    with tab1:
        st.subheader("Demographic Foundations")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Gender
        df["Gender"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[0])
        axes[0].set_title("Gender Distribution")
        axes[0].set_ylabel("")

        # Age (binned for better pie chart)
        age_bins = pd.cut(df["Age"], bins=[15, 18, 21, 24, 30, 60])
        age_bins.value_counts().plot.pie(autopct="%1.1f%%", ax=axes[1])
        axes[1].set_title("Age Group Distribution")
        axes[1].set_ylabel("")

        # Major
        df["Major"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[2])
        axes[2].set_title("Major Distribution")
        axes[2].set_ylabel("")

        st.pyplot(fig)

        st.markdown("""
        ### Representation
                    
        Gender: The student population is relatively balanced, with Males at 55.0% and Females at 45.0%.

        Age: The cohort is highly concentrated in early adulthood, with 18–21 year-olds (42.6%) [Undergraduates] and 21–24 year-olds (42.3%) [Graduates] making up nearly 85% of the data.

        Major: Students are distributed almost equally across six disciplines, with Psychology (17.4%) being the most common and Computer Science (15.9%) being the least common.
        """)

        st.markdown("""
        ### Significance
        The dataset represents a standard college students' population. The even distribution across majors ensures that performance comparisons between disciplines are statistically valid and not skewed by an overrepresentation of any single field.
                    """)


    with tab2:
       
        st.subheader("Disciplinary Performance Benchmarks")

        fig_box1, ax_box1 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Major', y='Previous_GPA', data=df, ax=ax_box1)
        plt.xticks(rotation=45)
        ax_box1.set_title('Previous GPA by Major')
        st.pyplot(fig_box1)

        fig_box2, ax_box2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Major', y='Attendance_Pct', data=df, ax=ax_box2)
        plt.xticks(rotation=45)
        ax_box2.set_title('Distribution of Attendance_PctAcross Different Majors')
        st.pyplot(fig_box2)
       
        st.markdown("""
        ### Representation
        Previous GPA: Median scores are consistent across all majors (ranging from 3.07 to 3.14), suggesting a uniform academic standard for incoming students regardless of their field.

        Attendance: Attendance percentages are similarly uniform, with all majors maintaining a median of approximately 85%.
        """)

        st.markdown("""
        ### Significance
The lack of significant variance in boxplot medians indicates that academic rigor and student commitment levels are standardized across the institution. There are no "outlier" majors where students perform significantly better or worse on average, pointing to a consistent educational environment.
       """)

    with tab3:

        st.subheader("Academic Achievement Distribution")
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(df['Final_CGPA'], kde=True, ax=ax_hist, color='purple')
        ax_hist.set_title('Overall CGPA Distribution')
        st.pyplot(fig_hist)

        st.markdown("""
        ### Representation
This histogram displays the frequency of Final CGPA scores.

The data follows a normal distribution centered at a mean of 3.27, with the majority of students achieving between 2.92 and 3.68.
        """)

        st.markdown("""
        ### Significance
The distribution suggests a healthy academic ecosystem where "grade inflation" is not rampant, but the vast majority of students are succeeding. The bell-shaped curve indicates that the grading system effectively differentiates between various levels of student performance.
       """)

    with tab4:

        st.subheader("Bivariate Performance Drivers")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x='Previous_GPA', y='Final_CGPA', data=df, ax=ax1, alpha=0.5)
        ax1.set_title('Previous GPA vs Final CGPA (Corr: 0.88)')
        st.pyplot(fig1)


        st.markdown("""
        ### Representation
Previous vs. Final GPA: Shows a very strong positive correlation (0.88). This is the most significant relationship in the dataset.           
        """)


    with tab5:

        st.subheader("Multivariate Correlation Matrix")
        fig4, ax4 = plt.subplots()
        correlation_matrix = df.select_dtypes(include=['number']).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", ax=ax4)
        ax4.set_title('Correlation Matrix Heatmap')
        st.pyplot(fig4)
 
        st.markdown("""
        ### Representation
The heatmap visualizes the strength of relationships between all numerical variables.

It highlights the Previous GPA (0.88) as the dominant factor, followed by Attendance (0.30) and Study Hours (0.23).
        """)

        st.markdown("""
        ### Significance
           The matrix reveals that lifestyle factors (sleep and socializing) are largely independent of academic success in this specific student body. This suggests that students may be effectively balancing their personal lives, or that these factors do not reach a threshold of disruption that impacts their final grades. Management should focus interventions on attendance and prior academic support to maximize student success.         
        
       """)

elif page == "Regression & Prediction":
# ---------------------------- Preprocessing for regression model -----------------------

    # - Change gender and major into numeric values
    df["Gender"] = df["Gender"].astype("category").cat.codes
    df["Major"] = df["Major"].astype("category").cat.codes

    # - Getting X and y for regression model
    from sklearn.model_selection import train_test_split
    X = df[['Gender',	'Age',	'Major',	'Attendance_Pct',	'Study_Hours_Per_Day',	'Previous_GPA',	'Sleep_Hours',	'Social_Hours_Week']]
    y = df["Final_CGPA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # - Train the regression model
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # - Evaluate the regression model
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    prediction = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    # - Create a dataframe to display the coefficients of the regression model
    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": lr.coef_
    })

# ---------------------------- Code for web app ------------------------------------------

    ## Step 03 - Regression Model
    st.subheader("03 Regression Model")

    tab2, tab3 = st.tabs(["Regression Model Accuracy", "Sweet Spot"])

    with tab2:
        st.subheader("Regression Model Accuracy")
        st.write(f"Root Mean Squared Error: {rmse}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R-squared Score: {r2}")
        st.subheader("Regression Coefficients")
        st.dataframe(coef_df)

        st.subheader("Actual vs Predicted Values (Scatter Plot)")
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, prediction)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted (Test Set)")

        # perfect prediction line
        plt.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()])

        st.pyplot(plt.gcf())

    with tab3:
        st.subheader("Sweet Spot")

    # Initialize session state
        if "study_hours" not in st.session_state:
            st.session_state.study_hours = 6
        if "sleep_hours" not in st.session_state:
            st.session_state.sleep_hours = 8
        if "social_hours" not in st.session_state:
            st.session_state.social_hours = 2

        total_hours = 24

        def get_max(key):
            """Max allowed for a slider = 24 minus the other two sliders' current values"""
            others = {k: st.session_state[k] for k in ["study_hours", "sleep_hours", "social_hours"] if k != key}
            return max(1, total_hours - sum(others.values()))

        study = st.slider(
            "Hours spent studying per day",
            0, get_max("study_hours"),
            st.session_state.study_hours,
            key="study_hours"
        )

        sleep = st.slider(
            "Hours spent sleeping per day",
            0, get_max("sleep_hours"),
            st.session_state.sleep_hours,
            key="sleep_hours"
        )

        social = st.slider(
            "Hours spent on social life per day",
            0, get_max("social_hours"),
            st.session_state.social_hours,
            key="social_hours"
        )

        remaining = total_hours - study - sleep - social
        st.info(f"Remaining unallocated hours: **{remaining}**")

        # Social hours per week (model expects weekly, slider is daily)
        social_weekly = social * 7

        # Use median/default values for other features
        gender_val = int(df["Gender"].median())
        age_val = df["Age"].median()
        major_val = int(df["Major"].median())
        attendance_val = df["Attendance_Pct"].median()
        prev_gpa_val = df["Previous_GPA"].median()

        # Build input array matching model's feature order
        input_data = pd.DataFrame([[
            gender_val,
            age_val,
            major_val,
            attendance_val,
            study,          # Study_Hours_Per_Day
            prev_gpa_val,
            sleep,          # Sleep_Hours
            social_weekly   # Social_Hours_Week
        ]], columns=['Gender', 'Age', 'Major', 'Attendance_Pct',
                    'Study_Hours_Per_Day', 'Previous_GPA', 'Sleep_Hours', 'Social_Hours_Week'])

        predicted_cgpa = min(lr.predict(input_data)[0],4.0)

        st.metric("📊 Predicted CGPA", f"{predicted_cgpa:.2f}")

# ---------------- Watermark ----------------
st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 15px;
        opacity: 0.25;
        font-size: 14px;
        color: gray;
        z-index: 1000;
    }
    </style>
    <div class="watermark">
        Coco Dai · Nadalia Jin · Solomon Kim · Aria Zhang
    </div>
    """,
    unsafe_allow_html=True
)
