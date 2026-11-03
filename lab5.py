import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Set Page Configuration
st.set_page_config(
    page_title="Single Layer Perceptron Laboratory - MAI511-2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
<style>
    .main-header { font-size: 32px; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 5px; }
    .sub-header { font-size: 18px; color: #555; text-align: center; margin-bottom: 25px; }
    .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 5px solid #1E88E5; }
    .stAlert { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Single Layer Perceptron Implementation Class
# ---------------------------------------------------------
class SingleLayerPerceptron:

  def __init__(self, learning_rate=0.1, epochs=20):
    self.lr = learning_rate
    self.epochs = epochs
    self.weights = None
    self.bias = None
    self.errors_per_epoch = []
    self.history = []

  def step_function(self, x):
    return np.where(x >= 0, 1, 0)

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0.0
    self.errors_per_epoch = []
    self.history = []

    for epoch in range(self.epochs):
      total_error = 0
      for i in range(n_samples):
        linear_output = np.dot(X[i], self.weights) + self.bias
        y_pred = self.step_function(linear_output)

        error = y[i] - y_pred
        update = self.lr * error

        self.weights += update * X[i]
        self.bias += update
        total_error += abs(error)

      self.errors_per_epoch.append(total_error)
      self.history.append({
          "epoch": epoch + 1,
          "weights": self.weights.copy(),
          "bias": self.bias,
          "total_error": total_error,
      })

      if total_error == 0:
        break

  def predict(self, X):
    linear_output = np.dot(X, self.weights) + self.bias
    return self.step_function(linear_output)

  def accuracy(self, y_true, y_pred):
    return np.mean(y_true == y_pred) * 100.0


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/000000/artificial-intelligence.png",
    width=80,
)
st.sidebar.title("Perceptron Lab Navigation")
st.sidebar.markdown("**Course**: MAI511-2 Advanced Machine Learning")
st.sidebar.markdown("**Register No Identifier**: `Aaron_101_Lab1`")

menu = st.sidebar.radio(
    "Select Lab Question / Module:",
    [
        "📌 Module Overview",
        "⚡ Q1: Logic Gates (AND, OR, NAND)",
        "🔀 Q2: XOR Gate Analysis & Limits",
        "🎓 Q3: Custom Student Pass/Fail Dataset",
        "📊 Expected Analysis Comparison Table",
    ],
)

# Binary Inputs Data
X_logic = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
logic_targets = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "XOR": np.array([0, 1, 1, 0]),
}

# ---------------------------------------------------------
# Module Overview
# ---------------------------------------------------------
if menu == "📌 Module Overview":
  st.markdown(
      '<div class="main-header">Single Layer Perceptron (SLP)'
      ' Simulator</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="sub-header">Interactive Exploration of Linear Separability,'
      ' Perceptron Learning Rule & Failure Modes</div>',
      unsafe_allow_html=True,
  )

  col1, col2 = st.columns([3, 2])
  with col1:
    st.subheader("📚 Learning Objectives")
    st.markdown("""
        - **Perceptron Learning Algorithm**: Understand weight and bias updates driven by error step signals:
          $$\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta (y - \\hat{y}) \\mathbf{x}, \\quad b \\leftarrow b + \\eta (y - \\hat{y})$$
        - **Linear Decision Boundaries**: Visualize hyperplanes separating binary classes ($w_1 x_1 + w_2 x_2 + b = 0$).
        - **Linear Separability vs. Non-Separability**: Contrast convergent logic gates (AND, OR, NAND) with non-convergent problems (XOR).
        - **Real-World Domain Application**: Apply SLP on a continuous dataset (Student Pass/Fail classification based on Study Hours and Attendance).
        """)

  with col2:
    st.subheader("⚙️ Perceptron Architecture")
    st.info("""
        **Inputs ($X_1, X_2$)** $\\rightarrow$ **Weighted Sum ($Z = w_1 x_1 + w_2 x_2 + b$)** $\\rightarrow$ **Step Activation Function** $\\rightarrow$ **Output ($\\hat{y} \\in \\{0, 1\\}$)**
        """)
    st.code(
        """
# Step Activation Function
def step_function(x):
    return np.where(x >= 0, 1, 0)
        """,
        language="python",
    )

# ---------------------------------------------------------
# Question 1: Logic Gates (AND, OR, NAND)
# ---------------------------------------------------------
elif menu == "⚡ Q1: Logic Gates (AND, OR, NAND)":
  st.header("⚡ Question 1: Single Layer Perceptron for Boolean Logic Gates")
  st.markdown(
      "Implement and simulate **AND**, **OR**, and **NAND** logic gates using"
      " the Rosenblatt Perceptron algorithm."
  )

  col_ctrl, col_display = st.columns([1, 2])

  with col_ctrl:
    st.subheader("Hyperparameters")
    gate_name = st.selectbox("Select Logic Gate:", ["AND", "OR", "NAND"])
    lr = st.slider("Learning Rate (η):", 0.01, 0.50, 0.10, step=0.01, key="q1_lr")
    epochs = st.slider("Max Epochs:", 5, 50, 20, key="q1_epochs")

    y_target = logic_targets[gate_name]

    model = SingleLayerPerceptron(learning_rate=lr, epochs=epochs)
    model.fit(X_logic, y_target)
    preds = model.predict(X_logic)
    acc = model.accuracy(y_target, preds)

    st.markdown("---")
    st.subheader("Training Metrics")
    st.metric(label="Classification Accuracy", value=f"{acc:.1f}%")
    st.metric(
        label="Learned Weights [w1, w2]", value=f"{np.round(model.weights, 2)}"
    )
    st.metric(label="Learned Bias (b)", value=f"{model.bias:.2f}")

  with col_display:
    st.subheader(f"Results for {gate_name} Gate")

    # Truth Table DataFrame
    df_res = pd.DataFrame({
        "Input X1": X_logic[:, 0],
        "Input X2": X_logic[:, 1],
        "Actual Target (y)": y_target,
        "Predicted (ŷ)": preds,
        "Match": np.where(y_target == preds, "✅ Correct", "❌ Incorrect"),
    })
    st.dataframe(df_res, use_container_width=True)

    # Decision Boundary Plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label in np.unique(y_target):
      ax.scatter(
          X_logic[y_target == label, 0],
          X_logic[y_target == label, 1],
          label=f"Target Output {label}",
          s=150,
          zorder=5,
      )

    w1, w2 = model.weights
    b = model.bias
    x1_vals = np.linspace(-0.5, 1.5, 100)

    if w2 != 0:
      x2_vals = -(w1 * x1_vals + b) / w2
      ax.plot(
          x1_vals, x2_vals, "k--", label="Decision Boundary Line", linewidth=2
      )
    else:
      ax.axvline(
          -b / w1, color="k", linestyle="--", label="Decision Boundary Line"
      )

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("Input $X_1$")
    ax.set_ylabel("Input $X_2$")
    ax.set_title(
        f"Decision Boundary for {gate_name} Gate",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)

# ---------------------------------------------------------
# Question 2: XOR Gate Analysis & Limits
# ---------------------------------------------------------
elif menu == "🔀 Q2: XOR Gate Analysis & Limits":
  st.header(
      "🔀 Question 2: Implementation and Analysis of SLP for XOR Logic"
  )
  st.write(
      "Demonstrating the convergence failure and non-linear separability of the"
      " **XOR Logic Gate**."
  )

  col_ctrl, col_display = st.columns([1, 2])

  with col_ctrl:
    st.subheader("Configuration")
    lr_xor = st.slider(
        "Learning Rate (η):", 0.01, 0.50, 0.10, step=0.01, key="q2_lr"
    )
    epochs_xor = st.slider("Max Epochs:", 5, 50, 20, key="q2_epochs")

    y_xor = logic_targets["XOR"]
    xor_model = SingleLayerPerceptron(learning_rate=lr_xor, epochs=epochs_xor)

    # Initialize weights and bias before manual tracking
    xor_model.weights = np.zeros(X_logic.shape[1])
    xor_model.bias = 0.0
    history_records = []

    for ep in range(epochs_xor):
      err_cnt = 0
      for i in range(len(X_logic)):
        l_out = np.dot(X_logic[i], xor_model.weights) + xor_model.bias
        p = xor_model.step_function(l_out)
        err = y_xor[i] - p
        xor_model.weights += lr_xor * err * X_logic[i]
        xor_model.bias += lr_xor * err
        err_cnt += abs(err)
      xor_model.errors_per_epoch.append(err_cnt)
      history_records.append({
          "Epoch": ep + 1,
          "Weights [w1, w2]": str(np.round(xor_model.weights, 2)),
          "Bias (b)": round(xor_model.bias, 2),
          "Total Errors": err_cnt,
      })

    xor_preds = xor_model.predict(X_logic)
    xor_acc = xor_model.accuracy(y_xor, xor_preds)

    st.markdown("---")
    st.error(f"**XOR Accuracy**: {xor_acc:.1f}% (Failed to converge)")
    st.metric("Final Weights", str(np.round(xor_model.weights, 2)))
    st.metric("Final Bias", f"{xor_model.bias:.2f}")

  with col_display:
    st.subheader("Epoch-wise Weight Updates & Error Log")
    df_hist = pd.DataFrame(history_records)
    st.dataframe(df_hist, height=200, use_container_width=True)

    st.subheader("Loss / Error Convergence Curve (AND vs XOR)")
    and_model = SingleLayerPerceptron(learning_rate=lr_xor, epochs=epochs_xor)
    and_model.fit(X_logic, logic_targets["AND"])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(
        range(1, len(and_model.errors_per_epoch) + 1),
        and_model.errors_per_epoch,
        "o-",
        label="AND Gate (Linearly Separable - Converges)",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        range(1, len(xor_model.errors_per_epoch) + 1),
        xor_model.errors_per_epoch,
        "s--",
        label="XOR Gate (Non-linearly Separable - Oscillates)",
        color="crimson",
        linewidth=2,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Classification Errors")
    ax.set_title(
        "Epoch vs Classification Error", fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, linestyle=":")
    st.pyplot(fig)

  st.markdown("---")
  st.subheader("📐 Mathematical Proof of Non-Linear Separability")
  st.markdown("""
    A Single Layer Perceptron classifies input $(x_1, x_2)$ using the condition $w_1 x_1 + w_2 x_2 + b \\ge 0$.  
    For the **XOR Gate**, the 4 truth table constraints are:
    1. $(0, 0) \\rightarrow 0 \\implies b < 0$
    2. $(0, 1) \\rightarrow 1 \\implies w_2 + b \\ge 0 \\implies w_2 \\ge -b > 0$
    3. $(1, 0) \\rightarrow 1 \\implies w_1 + b \\ge 0 \\implies w_1 \\ge -b > 0$
    4. $(1, 1) \\rightarrow 0 \\implies w_1 + w_2 + b < 0$

    **Contradiction Proof:**
    Adding inequalities (2) and (3) gives:
    $$(w_1 + b) + (w_2 + b) \\ge 0 \\implies w_1 + w_2 + 2b \\ge 0$$
    Since $b < 0$, subtracting $b$ from both sides yields:
    $$w_1 + w_2 + b \\ge -b > 0$$
    This directly contradicts inequality (4) ($w_1 + w_2 + b < 0$). **Thus, no straight line can separate XOR outputs.**
    """)

# ---------------------------------------------------------
# Question 3: Custom Student Pass/Fail Dataset
# ---------------------------------------------------------
elif menu == "🎓 Q3: Custom Student Pass/Fail Dataset":
  st.header("🎓 Question 3: Custom Domain Dataset Implementation")
  st.markdown(
      "Domain: **Academic Performance** — Predict student exam outcome"
      " (**Pass=1**, **Fail=0**) using **Study Hours** and **Attendance %**."
  )

  # Generate synthetic dataset
  np.random.seed(42)
  study_hours = np.array([
      1.5,
      2.0,
      5.0,
      6.5,
      1.0,
      7.0,
      4.0,
      3.5,
      8.0,
      2.5,
      6.0,
      1.2,
      7.5,
      3.0,
      8.5,
      4.5,
      5.5,
      2.2,
      9.0,
      0.5,
      6.2,
      3.8,
      7.8,
      1.8,
      8.2,
      4.8,
      5.2,
      2.8,
      9.5,
      1.1,
  ])
  attendance_pct = np.array([
      50,
      55,
      75,
      85,
      40,
      90,
      65,
      60,
      95,
      52,
      80,
      45,
      92,
      58,
      98,
      70,
      78,
      54,
      96,
      35,
      82,
      68,
      91,
      48,
      94,
      72,
      76,
      56,
      99,
      42,
  ])
  target_pass = np.where((study_hours >= 3.5) & (attendance_pct >= 60), 1, 0)

  df_custom = pd.DataFrame({
      "Study Hours": study_hours,
      "Attendance (%)": attendance_pct,
      "Pass Status": target_pass,
  })

  col_ctrl, col_display = st.columns([1, 2])

  with col_ctrl:
    st.subheader("Training Parameters")
    lr_cust = st.slider(
        "Learning Rate (η):", 0.01, 0.50, 0.10, step=0.01, key="q3_lr"
    )
    epochs_cust = st.slider("Max Epochs:", 10, 100, 50, key="q3_epochs")

    # Feature Standardization
    X_custom = df_custom[["Study Hours", "Attendance (%)"]].values
    y_custom = df_custom["Pass Status"].values
    X_mean = np.mean(X_custom, axis=0)
    X_std = np.std(X_custom, axis=0)
    X_scaled = (X_custom - X_mean) / X_std

    c_slp = SingleLayerPerceptron(learning_rate=lr_cust, epochs=epochs_cust)
    c_slp.fit(X_scaled, y_custom)
    c_preds = c_slp.predict(X_scaled)
    c_acc = c_slp.accuracy(y_custom, c_preds)

    st.markdown("---")
    st.success(f"**Classification Accuracy**: {c_acc:.2f}%")
    st.metric("Learned Weights [w1, w2]", str(np.round(c_slp.weights, 4)))
    st.metric("Learned Bias (b)", f"{c_slp.bias:.4f}")

    st.markdown("---")
    st.subheader("🔮 Single Student Predictor")
    in_hours = st.number_input(
        "Study Hours/Day:", min_value=0.0, max_value=12.0, value=4.5, step=0.5
    )
    in_att = st.number_input(
        "Attendance (%):", min_value=0.0, max_value=100.0, value=75.0, step=5.0
    )

    sample_scaled = (np.array([[in_hours, in_att]]) - X_mean) / X_std
    pred_val = c_slp.predict(sample_scaled)[0]
    if pred_val == 1:
      st.success("🎉 Predicted Outcome: **PASS (1)**")
    else:
      st.error("⚠️ Predicted Outcome: **FAIL (0)**")

  with col_display:
    st.subheader("Dataset Sample & Perceptron Decision Boundary")
    st.dataframe(df_custom.head(8), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df_custom,
        x="Study Hours",
        y="Attendance (%)",
        hue="Pass Status",
        style="Pass Status",
        s=120,
        palette=["red", "green"],
        ax=ax,
    )

    # Reconstruct decision line in unscaled feature space
    w1, w2 = c_slp.weights
    b = c_slp.bias
    x1_raw = np.linspace(0, 10, 100)
    x1_scaled = (x1_raw - X_mean[0]) / X_std[0]
    x2_scaled = -(w1 * x1_scaled + b) / w2
    x2_raw = (x2_scaled * X_std[1]) + X_mean[1]

    ax.plot(
        x1_raw,
        x2_raw,
        "b--",
        label="SLP Linear Decision Boundary",
        linewidth=2,
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(30, 105)
    ax.set_title(
        "Student Pass/Fail Decision Boundary", fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)

# ---------------------------------------------------------
# Question 4: Expected Analysis Comparison Table
# ---------------------------------------------------------
elif menu == "📊 Expected Analysis Comparison Table":
  st.header("📊 Expected Analysis & Comparison Table")
  st.markdown(
      "Summary of Perceptron performance across all tested logic gates as"
      " specified in the lab evaluation requirements."
  )

  res_list = []
  for g_name, y_t in logic_targets.items():
    m = SingleLayerPerceptron(learning_rate=0.1, epochs=20)
    if g_name == "XOR":
      m.weights = np.zeros(2)
      m.bias = 0.0
      for _ in range(20):
        for i in range(4):
          p = m.step_function(np.dot(X_logic[i], m.weights) + m.bias)
          err = y_t[i] - p
          m.weights += 0.1 * err * X_logic[i]
          m.bias += 0.1 * err
    else:
      m.fit(X_logic, y_t)

    p_out = m.predict(X_logic)
    acc_val = m.accuracy(y_t, p_out)

    is_sep = "No" if g_name == "XOR" else "Yes"
    outcome = (
        "Does not converge to a perfect solution"
        if g_name == "XOR"
        else "Converges"
    )

    res_list.append({
        "Logic Gate": g_name,
        "Linearly Separable": is_sep,
        "SLP Learning Outcome": outcome,
        "Accuracy (%)": f"{acc_val:.0f}%",
        "Final Weights [w1, w2]": str(np.round(m.weights, 2)),
        "Final Bias (b)": f"{m.bias:.2f}",
    })

  df_comp = pd.DataFrame(res_list)
  st.table(df_comp)

  st.success("""
    ### 💡 Key Takeaways:
    1. **Single Layer Perceptrons** can accurately solve any logic gate or dataset that is **linearly separable** (AND, OR, NAND).
    2. **XOR** fails because the outputs cannot be separated by a single 1D hyper-plane ($w_1 x_1 + w_2 x_2 + b = 0$).
    3. Solving non-linearly separable problems requires **Multi-Layer Perceptrons (MLPs)** with hidden layers and non-linear activation functions (e.g., Sigmoid, ReLU).
    """)