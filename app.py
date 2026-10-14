import io
import os
import random
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "human_activity_cnn_lstm.keras")
SCALER_PATH = os.path.join(BASE, "models", "human_activity_scaler.pkl")
DEMO_PATH = os.path.join(BASE, "models", "demo_samples.npz")

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]
ICONS = {"WALKING": "🚶", "WALKING_UPSTAIRS": "🪜", "WALKING_DOWNSTAIRS": "⬇️",
         "SITTING": "🪑", "STANDING": "🧍", "LAYING": "🛌"}
COLORS = {"WALKING": "#22c55e", "WALKING_UPSTAIRS": "#3b82f6",
          "WALKING_DOWNSTAIRS": "#a855f7", "SITTING": "#f59e0b",
          "STANDING": "#06b6d4", "LAYING": "#ef4444"}
CH_ACC = ["acc_x", "acc_y", "acc_z"]
CH_GYR = ["gyro_x", "gyro_y", "gyro_z"]
CHANNELS = CH_ACC + CH_GYR
COLOR_SCALE = alt.Scale(domain=ACTIVITIES, range=[COLORS[a] for a in ACTIVITIES])

st.set_page_config(page_title="Human Activity Recognition", page_icon="🏃", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.hero {background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#ec4899 100%);
       padding: 30px 36px; border-radius: 22px; color: white; margin-bottom: 20px;
       box-shadow: 0 10px 30px rgba(99,102,241,.35);}
.hero h1 {margin: 0; font-size: 2.3rem; color: white;}
.hero p {margin: 6px 0 0 0; font-size: 1.05rem; opacity: .92;}
.badge {display:inline-block; background: rgba(255,255,255,.2); padding: 4px 14px;
        border-radius: 999px; margin: 12px 8px 0 0; font-size: .85rem;}
.result-card {border-radius: 20px; padding: 26px; color: white; text-align: center;
              box-shadow: 0 8px 24px rgba(0,0,0,.35);}
.result-card .emoji {font-size: 4rem; line-height: 1.1;}
.result-card .name {font-size: 1.7rem; font-weight: 800; letter-spacing: 1px; margin-top: 6px;}
.result-card .conf {font-size: 1.1rem; margin-top: 8px; opacity: .95;}
.bar-bg {background: rgba(255,255,255,.28); border-radius: 999px; height: 12px; margin-top: 10px;}
.bar-fg {background: white; border-radius: 999px; height: 12px;}
.pill-ok {background:#14532d; color:#86efac; padding:10px 16px; border-radius:12px; margin-top:14px; text-align:center; font-weight:600;}
.pill-bad {background:#7f1d1d; color:#fca5a5; padding:10px 16px; border-radius:12px; margin-top:14px; text-align:center; font-weight:600;}
.actual {margin-top: 14px; padding: 10px 16px; border-radius: 12px; background: rgba(148,163,184,.15); text-align:center;}
.stat {background: rgba(148,163,184,.12); border-radius: 16px; padding: 18px; text-align:center;}
.stat .v {font-size: 2rem; font-weight: 800;}
.stat .l {opacity: .75; font-size: .9rem;}
.footer {text-align:center; opacity:.6; margin-top: 30px; font-size:.85rem;}
button[data-baseweb="tab"] {font-size: 1.05rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏃 Human Activity Recognition</h1>
  <p>Smartphone accelerometer + gyroscope signals → 1D-CNN + LSTM → six everyday activities</p>
  <span class="badge">🧠 79,878 parameters</span>
  <span class="badge">🎯 92.06% test accuracy</span>
  <span class="badge">📱 UCI HAR dataset</span>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model
def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dropout, Dense
    return Sequential([
        Input(shape=(128, 6)),
        Conv1D(64, 3, activation="relu"), MaxPooling1D(2),
        Conv1D(128, 3, activation="relu"), MaxPooling1D(2),
        LSTM(64), Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(6, activation="softmax"),
    ])


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
    except Exception:
        model = build_model()
        model.load_weights(MODEL_PATH)
    return model, scaler


@st.cache_data
def load_demo():
    d = np.load(DEMO_PATH)
    return d["X"], d["y"]


for p in (MODEL_PATH, SCALER_PATH, DEMO_PATH):
    if not os.path.exists(p):
        st.error(f"File missing: {p}\n\nPut the 3 files from Colab inside the 'models' folder.")
        st.stop()

try:
    with st.spinner("Loading model (first time takes 30-60 seconds)..."):
        model, scaler = load_artifacts()
except Exception as e:
    st.error("Model could not be loaded. Error details below:")
    st.exception(e)
    st.stop()

Xd, yd = load_demo()


def predict_batch(windows):
    scaled = scaler.transform(windows.reshape(-1, 6)).reshape(-1, 128, 6)
    return model.predict(scaled, verbose=0)


def predict(window):
    return predict_batch(window[np.newaxis])[0]


# ------------------------------------------------------------------ widgets
def result_card(probs, actual=None):
    pred = int(np.argmax(probs))
    name = ACTIVITIES[pred]
    conf = probs[pred] * 100
    c = COLORS[name]
    st.markdown(f"""
    <div class="result-card" style="background: linear-gradient(135deg,{c},{c}aa);">
      <div class="emoji">{ICONS[name]}</div>
      <div class="name">{name.replace('_', ' ')}</div>
      <div class="conf">Confidence: <b>{conf:.2f}%</b></div>
      <div class="bar-bg"><div class="bar-fg" style="width:{conf:.1f}%"></div></div>
    </div>""", unsafe_allow_html=True)
    if actual is not None:
        an = ACTIVITIES[actual]
        st.markdown(f'<div class="actual">Actual activity: <b>{ICONS[an]} {an.replace("_", " ")}</b></div>',
                    unsafe_allow_html=True)
        if actual == pred:
            st.markdown('<div class="pill-ok">✅ Correct prediction</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pill-bad">❌ Wrong prediction</div>', unsafe_allow_html=True)


def prob_chart(probs):
    df = pd.DataFrame({"Activity": ACTIVITIES, "Probability": probs * 100})
    chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=6, size=26)
             .encode(x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 100]), title="Probability (%)"),
                     y=alt.Y("Activity:N", sort=ACTIVITIES, title=None),
                     color=alt.Color("Activity:N", scale=COLOR_SCALE, legend=None),
                     tooltip=["Activity", alt.Tooltip("Probability:Q", format=".2f")])
             .properties(height=280))
    st.altair_chart(chart, use_container_width=True)


def signal_chart(window, cols, title):
    df = pd.DataFrame(window, columns=CHANNELS)[cols]
    df["Time step"] = np.arange(len(df))
    long = df.melt("Time step", var_name="Channel", value_name="Value")
    chart = (alt.Chart(long).mark_line(strokeWidth=2)
             .encode(x="Time step:Q", y="Value:Q",
                     color=alt.Color("Channel:N", scale=alt.Scale(
                         range=["#f472b6", "#38bdf8", "#facc15"])),
                     tooltip=["Channel", "Time step", alt.Tooltip("Value:Q", format=".3f")])
             .properties(height=240, title=title))
    st.altair_chart(chart, use_container_width=True)


def show_signals(window):
    c1, c2 = st.columns(2)
    with c1:
        signal_chart(window, CH_ACC, "📈 Accelerometer (X, Y, Z)")
    with c2:
        signal_chart(window, CH_GYR, "🌀 Gyroscope (X, Y, Z)")


def stat(value, label):
    return f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div></div>'


# ------------------------------------------------------------------ tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Prediction", "🧪 Test All Samples",
                                  "📊 Model Performance", "ℹ️ About"])

# ---------------- Tab 1: live prediction
with tab1:
    mode = st.radio("Choose input", ["Demo sample", "Upload CSV"], horizontal=True,
                    label_visibility="collapsed")
    if mode == "Demo sample":
        st.session_state.setdefault("act", ACTIVITIES[0])
        st.session_state.setdefault("k", 1)

        def randomize():
            st.session_state.act = random.choice(ACTIVITIES)
            st.session_state.k = random.randint(1, 10)

        s1, s2, s3 = st.columns([2, 2, 1])
        act = s1.selectbox("Activity to test", ACTIVITIES, key="act",
                           format_func=lambda a: f"{ICONS[a]}  {a.replace('_', ' ')}")
        idxs = np.where(yd == ACTIVITIES.index(act))[0]
        k = s2.slider("Sample number", 1, len(idxs), key="k")
        s3.write("")
        s3.button("🎲 Random", on_click=randomize, use_container_width=True)

        i = idxs[min(k, len(idxs)) - 1]
        window = Xd[i]
        probs = predict(window)
        left, right = st.columns([1, 2])
        with left:
            result_card(probs, actual=int(yd[i]))
        with right:
            st.markdown("##### Class probabilities")
            prob_chart(probs)
        st.markdown("##### Sensor signals (128 time steps = 2.56 seconds)")
        show_signals(window)
        buf = io.StringIO()
        pd.DataFrame(window, columns=CHANNELS).to_csv(buf, index=False)
        st.download_button("⬇️ Download this window as CSV", buf.getvalue(),
                           file_name=f"sample_{act}_{k}.csv", mime="text/csv")
    else:
        st.info("Upload a CSV with **128 rows × 6 columns**: "
                "`acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z` (header optional).")
        f = st.file_uploader("CSV file", type=["csv"])
        if f is not None:
            try:
                df = pd.read_csv(f, header=None).apply(pd.to_numeric, errors="coerce").dropna()
                if df.shape[1] != 6:
                    st.error(f"Expected 6 columns, got {df.shape[1]}.")
                elif df.shape[0] < 128:
                    st.error(f"Need at least 128 rows, got {df.shape[0]}.")
                else:
                    window = df.values[:128].astype("float32")
                    probs = predict(window)
                    left, right = st.columns([1, 2])
                    with left:
                        result_card(probs)
                    with right:
                        st.markdown("##### Class probabilities")
                        prob_chart(probs)
                    show_signals(window)
            except Exception as e:
                st.error(f"Could not read file: {e}")

# ---------------- Tab 2: test all samples
with tab2:
    st.markdown(f"Run the model on all **{len(Xd)} demo samples** at once and see how many it gets right.")
    if st.button("▶ Run test on all samples", type="primary"):
        P = predict_batch(Xd)
        pred = P.argmax(axis=1)
        ok = pred == yd
        st.session_state["test_all"] = pd.DataFrame({
            "Actual": [ACTIVITIES[a] for a in yd],
            "Predicted": [ACTIVITIES[a] for a in pred],
            "Confidence %": (P.max(axis=1) * 100).round(2),
            "Result": ["✅" if o else "❌" for o in ok]})
    res = st.session_state.get("test_all")
    if res is not None:
        correct = int((res["Result"] == "✅").sum())
        total = len(res)
        c1, c2, c3 = st.columns(3)
        c1.markdown(stat(f"{correct}/{total}", "Correct predictions"), unsafe_allow_html=True)
        c2.markdown(stat(f"{correct / total * 100:.1f}%", "Accuracy on these samples"), unsafe_allow_html=True)
        c3.markdown(stat(total - correct, "Wrong predictions"), unsafe_allow_html=True)
        st.write("")
        per = (res.assign(ok=res["Result"] == "✅").groupby("Actual")["ok"].mean() * 100).reindex(ACTIVITIES)
        pdf = pd.DataFrame({"Activity": per.index, "Accuracy": per.values})
        chart = (alt.Chart(pdf).mark_bar(cornerRadiusEnd=6)
                 .encode(x=alt.X("Activity:N", sort=ACTIVITIES, axis=alt.Axis(labelAngle=-30), title=None),
                         y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
                         color=alt.Color("Activity:N", scale=COLOR_SCALE, legend=None),
                         tooltip=["Activity", alt.Tooltip("Accuracy:Q", format=".1f")])
                 .properties(height=300, title="Accuracy per activity"))
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(res, use_container_width=True, hide_index=True)

# ---------------- Tab 3: performance (from the project report)
with tab3:
    st.caption("Results from the project report (UCI HAR test set, 2,947 windows). "
               "Your newly trained model may differ by a small amount.")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat("92.06%", "Test accuracy"), unsafe_allow_html=True)
    c2.markdown(stat("0.92", "Macro F1-score"), unsafe_allow_html=True)
    c3.markdown(stat("79,878", "Trainable parameters"), unsafe_allow_html=True)
    c4.markdown(stat("2,947", "Test samples"), unsafe_allow_html=True)
    st.write("")

    report = pd.DataFrame({
        "Activity": ACTIVITIES,
        "Precision": [0.99, 0.98, 0.89, 0.83, 0.90, 0.94],
        "Recall": [0.95, 0.94, 0.98, 0.92, 0.93, 0.82],
        "F1-score": [0.97, 0.96, 0.93, 0.87, 0.92, 0.88],
        "Support": [496, 471, 420, 491, 532, 537]})
    long = report.melt(id_vars=["Activity", "Support"], value_vars=["Precision", "Recall", "F1-score"],
                       var_name="Metric", value_name="Score")
    g1, g2 = st.columns([3, 2])
    with g1:
        chart = (alt.Chart(long).mark_bar(cornerRadiusEnd=3)
                 .encode(x=alt.X("Metric:N", axis=None, sort=["Precision", "Recall", "F1-score"]),
                         y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                         color=alt.Color("Metric:N", scale=alt.Scale(
                             domain=["Precision", "Recall", "F1-score"],
                             range=["#818cf8", "#34d399", "#fbbf24"])),
                         column=alt.Column("Activity:N", sort=ACTIVITIES,
                                           header=alt.Header(labelAngle=-45, labelAlign="right",
                                                             labelFontSize=10, title=None)),
                         tooltip=["Activity", "Metric", "Score"])
                 .properties(width=48, height=280, title="Precision / Recall / F1 per activity"))
        st.altair_chart(chart, use_container_width=False)
    with g2:
        st.markdown("##### Classification report")
        st.dataframe(report, hide_index=True, use_container_width=True)

    cm = np.array([[469, 4, 23, 0, 0, 0], [1, 443, 27, 0, 0, 0], [1, 6, 413, 0, 0, 0],
                   [0, 0, 0, 452, 21, 18], [1, 1, 0, 28, 494, 8], [0, 0, 1, 63, 31, 442]])
    cmdf = pd.DataFrame([(ACTIVITIES[i], ACTIVITIES[j], int(cm[i, j]))
                         for i in range(6) for j in range(6)], columns=["Actual", "Predicted", "Count"])
    base = alt.Chart(cmdf).encode(
        x=alt.X("Predicted:N", sort=ACTIVITIES, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Actual:N", sort=ACTIVITIES))
    heat = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="magma"), legend=None))
    text = base.mark_text(fontSize=13, fontWeight="bold").encode(
        text="Count:Q", color=alt.condition(alt.datum.Count > 250, alt.value("black"), alt.value("white")))
    st.markdown("##### Confusion matrix")
    st.altair_chart((heat + text).properties(height=420), use_container_width=True)
    st.info("💡 Most errors happen **inside** a group: walking variants with each other, and "
            "sitting / standing / laying with each other. Only 3 errors mix a moving and a static activity.")

# ---------------- Tab 4: about
with tab4:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("### How it works")
        st.markdown("""
1. 📱 **Sensors** – accelerometer and gyroscope record 128 readings (2.56 s) on 6 channels
2. ⚖️ **Normalisation** – StandardScaler (fitted on training data only)
3. 🔍 **1D-CNN** – learns local movement patterns
4. ⏱️ **LSTM** – learns how the patterns change over time
5. 🎯 **Softmax** – probability for each of the 6 activities
        """)
        st.markdown("### Activities")
        for a in ACTIVITIES:
            st.markdown(f"<span style='color:{COLORS[a]};font-weight:700'>{ICONS[a]} {a.replace('_', ' ')}</span>",
                        unsafe_allow_html=True)
    with a2:
        st.markdown("### Model architecture")
        st.dataframe(pd.DataFrame({
            "Layer": ["Input", "Conv1D (64, k=3)", "MaxPooling1D (2)", "Conv1D (128, k=3)",
                      "MaxPooling1D (2)", "LSTM (64)", "Dropout (0.5)", "Dense (64, ReLU)", "Dense (6, Softmax)"],
            "Output shape": ["128 × 6", "126 × 64", "63 × 64", "61 × 128", "30 × 128", "64", "64", "64", "6"]}),
            hide_index=True, use_container_width=True)
        st.warning("⚠️ Decision-support tool only. Not for medical or safety decisions.")

st.markdown('<div class="footer">Deep Learning Mini Project · KGiSL Institute of Technology · '
            'Activity Recognition using 1D-CNN + LSTM</div>', unsafe_allow_html=True)
