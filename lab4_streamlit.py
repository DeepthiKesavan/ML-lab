import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

st.title("Lab 4: Dimensionality Reduction")
st.write("Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Independent Component Analysis (ICA)")

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Original Dataset")
    st.dataframe(df.head())

    target_column = st.selectbox(
        "Select the target column for LDA (if available)",
        ["None"] + list(df.columns)
    )

    method = st.selectbox(
        "Choose a method",
        ["PCA", "LDA", "ICA"]
    )

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        st.error("The dataset must contain numeric features.")
    else:
        if method == "LDA" and target_column != "None":
            X = df.drop(columns=[target_column])
            X = X.select_dtypes(include=["number"])
            y = df[target_column]

            if X.empty:
                st.error("No numeric feature columns found.")
            elif y.nunique() < 2:
                st.error("LDA requires at least two classes in the target column.")
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                max_components = min(X.shape[1], y.nunique() - 1)
                n_components = st.slider(
                    "Number of components",
                    min_value=1,
                    max_value=max_components,
                    value=1
                )

                model = LinearDiscriminantAnalysis(n_components=n_components)
                transformed = model.fit_transform(X_scaled, y)

                result = pd.DataFrame(
                    transformed,
                    columns=[f"LD{i+1}" for i in range(transformed.shape[1])]
                )

                st.subheader("LDA Transformed Data")
                st.dataframe(result.head())

                if transformed.shape[1] >= 2:
                    fig, ax = plt.subplots()
                    scatter = ax.scatter(
                        transformed[:, 0],
                        transformed[:, 1],
                        c=pd.factorize(y)[0]
                    )
                    ax.set_xlabel("LD1")
                    ax.set_ylabel("LD2")
                    ax.set_title("LDA Visualization")
                    st.pyplot(fig)

        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_df)

            max_components = min(X_scaled.shape[1], X_scaled.shape[0])
            n_components = st.slider(
                "Number of components",
                min_value=1,
                max_value=max_components,
                value=min(2, max_components)
            )

            if method == "PCA":
                model = PCA(n_components=n_components)
                transformed = model.fit_transform(X_scaled)

                st.write("Explained variance ratio:")
                st.write(model.explained_variance_ratio_)

                st.write(
                    "Total explained variance:",
                    round(model.explained_variance_ratio_.sum(), 4)
                )

            elif method == "ICA":
                model = FastICA(
                    n_components=n_components,
                    random_state=42,
                    max_iter=1000,
                    tol=0.01
                )
                transformed = model.fit_transform(X_scaled)

            result = pd.DataFrame(
                transformed,
                columns=[f"Component{i+1}" for i in range(transformed.shape[1])]
            )

            st.subheader(f"{method} Transformed Data")
            st.dataframe(result.head())

            if transformed.shape[1] >= 2:
                fig, ax = plt.subplots()
                ax.scatter(transformed[:, 0], transformed[:, 1])
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_title(f"{method} Visualization")
                st.pyplot(fig)

            st.download_button(
                "Download transformed data",
                result.to_csv(index=False),
                file_name=f"{method.lower()}_transformed.csv",
                mime="text/csv"
            )
else:
    st.info("Upload a CSV file to begin.")