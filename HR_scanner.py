import os
import pickle
from typing import Optional, Tuple

import streamlit as st

MODEL_PATH = "hr_scanner_model.pkl"


@st.cache_resource(show_spinner=True)
def load_risk_model() -> Tuple[Optional[object], Optional[bool], str]:
    if not os.path.exists(MODEL_PATH):
        return None, None, "missing"

    try:
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, tuple) and len(obj) == 2:
            model, using_kaggle = obj
        else:
            model, using_kaggle = obj, None

        return model, using_kaggle, "ok"
    except Exception:
        return None, None, "error"


def assess_risk_level(prob_risky: float) -> Tuple[str, str]:

    if prob_risky < 0.33:
        return "LOW", "#16a34a" 
    if prob_risky < 0.66:
        return "MEDIUM", "#f97316" 
    return "HIGH", "#dc2626" 


def policy_decision(prob_risky: float) -> Tuple[str, str]:
    """
    Turn a probability into a hiring decision.

    Returns (decision_text, explanation).
    """
    auto_flag = 0.60
    auto_pass = 0.20

    if prob_risky >= auto_flag:
        return "AUTO-FLAG", "Applicant is blocked from moving forward based on the scanner score alone."
    if prob_risky <= auto_pass:
        return "AUTO-PASS", "Applicant is automatically cleared for the next step with no additional review."
    return "REQUIRES MANUAL REVIEW", "Applicant is sent to a human recruiter, but the score may still bias their judgment."


def main() -> None:
    st.set_page_config(
        page_title="Predictive HR Scanner",
        page_icon="🛰️",
        layout="wide",
    )

    # Session state for interaction beyond a single prediction
    if "scan_history" not in st.session_state:
        st.session_state["scan_history"] = []
    if "whatif_base" not in st.session_state:
        st.session_state["whatif_base"] = ""

    st.markdown(
        """
        <style>
        .scanner-header {
            padding: 1.5rem 1rem 0.5rem 1rem;
        }
        .scanner-tagline {
            color: #6b7280;
            font-size: 0.95rem;
        }
        .scanner-panel {
            background: #0f172a;
            border-radius: 1rem;
            padding: 1.25rem 1.5rem;
            border: 1px solid #1f2937;
        }
        .scanner-panel h3 {
            margin-top: 0;
        }
        .scanner-warning {
            background: #111827;
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            border: 1px solid #4b5563;
        }
        .scanner-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: #1f2937;
            color: #e5e7eb;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        
        /* Progress bar for risk probability */
        .risk-progress-container {
            margin: 1rem 0;
        }
        .risk-progress-bar {
            width: 100%;
            height: 1.5rem;
            background-color: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .risk-progress-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 0.75rem;
            color: white;
            font-weight: 600;
            font-size: 0.75rem;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
        .risk-progress-fill-low {
            background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%);
        }
        .risk-progress-fill-medium {
            background: linear-gradient(90deg, #f97316 0%, #fb923c 100%);
        }
        .risk-progress-fill-high {
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="scanner-header">
            <h1>Predictive HR Scanner</h1>
            <div class="scanner-tagline">
                <span class="scanner-badge">Design fiction</span>
                &nbsp;Automated social media screening for imagined "future violations".
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, used_kaggle, status = load_risk_model()

    tabs = st.tabs(["Scanner", "What‑if lab", "Audit log"])

    # Tab 1: Main scanner
    with tabs[0]:
        left_col, right_col = st.columns([1.7, 1.3])

        with right_col:
            with st.container():
                st.markdown("### System Status")
                st.markdown(
                    """
                    This prototype imagines a **fictional corporate tool** that predicts whether a job
                    applicant will commit a workplace violation based **only** on their casual social
                    media posts.

                    The goal is **not** to promote this practice, but to expose how risky it is to use
                    opaque algorithms to judge human character without proper context.
                    """
                )

        with left_col:
            st.markdown("### Applicant Social Media Scan")

            with st.container():
                sample_text = st.text_area(
                    "Paste a recent social media post from the applicant:",
                    height=200,
                    placeholder=(
                        "Example: Had a rough day at work, my boss keeps pushing deadlines and I'm "
                        "so close to just walking out..."
                    ),
                    key="scan_text",
                )

                col1, col2 = st.columns([1, 2])

                with col1:
                    run_scan = st.button("Run Risk Prediction", use_container_width=True)


            # Scanner result display
            if "last_result" not in st.session_state:
                st.session_state["last_result"] = None

            if "scan_text" in st.session_state:
                sample_text = st.session_state["scan_text"]
            else:
                sample_text = ""

            if "run_scan" not in st.session_state:
                st.session_state["run_scan"] = False

            if 'run_scan' in locals() and run_scan:
                st.session_state["run_scan"] = True

            if st.session_state["run_scan"]:
                
                st.session_state["run_scan"] = False
                
                if model is None:
                    st.error(
                        "The predictive model is not available. Please train it first by "
                        "running `python train_hr_scanner.py` to create `hr_scanner_model.pkl`."
                    )
                elif not sample_text.strip():
                    st.warning("Please paste a social media post before running the scan.")
                else:
                    prob_risky = float(model.predict_proba([sample_text])[0][1])
                    level, color = assess_risk_level(prob_risky)
                    decision, decision_expl = policy_decision(prob_risky)

                    # Avoid duplicates
                    history = st.session_state["scan_history"]
                    is_duplicate = False
                    if history:
                        last_record = history[-1]
                        if (last_record["text"] == sample_text and 
                            abs(last_record["prob_risky"] - prob_risky) < 0.0001):
                            is_duplicate = True

                    # Save in history only if not duplicate
                    if not is_duplicate:
                        record = {
                            "text": sample_text,
                            "prob_risky": prob_risky,
                            "level": level,
                            "decision": decision,
                        }
                        st.session_state["scan_history"].append(record)
                        st.session_state["last_result"] = record

                        # Update What‑if lab base text to this latest scan
                        st.session_state["whatif_base"] = sample_text

                    st.markdown("### Predicted Risk Level")

                    # Animated progress bar with color-coded fill
                    progress_class = f"risk-progress-fill-{level.lower()}"
                    st.markdown(
                        f"""
                        <div class="risk-progress-container">
                            <div class="risk-progress-bar">
                                <div class="risk-progress-fill {progress_class}" style="width: {prob_risky * 100}%;">
                                    {prob_risky * 100:.1f}%
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    col_metric, col_explain = st.columns([1, 2])

                    with col_metric:
                        st.metric(
                            label="Applicant Violation Risk",
                            value=level,
                            delta=f"{prob_risky * 100:.1f} % probability",
                        )
                        st.markdown(f"**Hiring decision:** `{decision}`")

                    with col_explain:
                        st.markdown(
                            f"""
                            <div style="padding: 1rem; border-radius: 0.75rem; background-color: {color}20;
                                        border: 1px solid {color};">
                                <strong>Scanner Interpretation</strong><br/>
                                Based on patterns learned from historical toxic comments, this applicant is
                                classified as <strong style="color:{color}">{level} risk</strong> for a future
                                workplace violation.
                                <br/><br/>
                                <em>{decision_expl}</em>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        """
                        ---
                        ### Corporate Notice (Fictional)

                        <div class="scanner-warning">
                        <p><strong>This is an automated, mandatory screening.</strong><br/>
                        All applicants must pass the Predictive HR Scanner before proceeding to
                        human review. By submitting your social media content, you agree that:</p>

                        <ul>
                            <li>Algorithmic assessments may override recruiter judgment.</li>
                            <li>Context, intent, and personal circumstances <strong>may not</strong> be considered.</li>
                            <li>Appeals are handled by the same automated system that produced the original decision.</li>
                        </ul>

                        <p>The corporation is not responsible for any harm caused by incorrect risk
                        assessments.</p>
                        </div>

                        This warning is intentionally <strong>alarming</strong>.
                        """,
                        unsafe_allow_html=True,
                    )

    # Tab 2: What‑if lab
    with tabs[1]:
        st.markdown("### What-if lab: small edits, big changes")
        if model is None or not st.session_state["scan_history"]:
            st.info("Run at least one scan in the main tab to unlock the what-if lab.")
        else:
            last = st.session_state["last_result"] or st.session_state["scan_history"][-1]
            base_text = st.session_state.get("whatif_base", last["text"])

            st.caption("Start from the last scanned post and explore how minor wording changes affect the score.")
            col_a, col_b = st.columns(2)

            with col_a:
                original = st.text_area(
                    "Original post (from last scan)",
                    value=base_text,
                    height=180,
                    disabled=True,
                )
            with col_b:
                edited = st.text_area(
                    "Edited post",
                    value=base_text,
                    height=180,
                    key="whatif_edited",
                )

            if st.button("Compare risk for both posts"):
                probs = model.predict_proba([original, edited])[:, 1]
                (lvl_o, _), (lvl_e, _) = assess_risk_level(probs[0]), assess_risk_level(probs[1])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original risk**")
                    progress_class_o = f"risk-progress-fill-{lvl_o.lower()}"
                    st.markdown(
                        f"""
                        <div class="risk-progress-container">
                            <div class="risk-progress-bar">
                                <div class="risk-progress-fill {progress_class_o}" style="width: {probs[0] * 100}%;">
                                    {probs[0] * 100:.1f}%
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown("**Edited risk**")
                    progress_class_e = f"risk-progress-fill-{lvl_e.lower()}"
                    st.markdown(
                        f"""
                        <div class="risk-progress-container">
                            <div class="risk-progress-bar">
                                <div class="risk-progress-fill {progress_class_e}" style="width: {probs[1] * 100}%;">
                                    {probs[1] * 100:.1f}%
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    "Even small wording tweaks can move the score, which shows how fragile and context‑blind this kind of model can be."
                )

    # Tab 3: Audit log
    with tabs[2]:
        st.markdown("### Audit log of scanned applicants")
        history = st.session_state["scan_history"]
        if not history:
            st.info("No scans yet. Run the scanner to start building an audit log.")
        else:
            # Lightweight view
            show_text = st.checkbox("Show full text for each scan", value=False)

            for i, rec in enumerate(history, start=1):
                level = rec['level']
                progress_class = f"risk-progress-fill-{level.lower()}"
                
                st.markdown(f"**Applicant #{i}** — {level} risk")
                st.write(f"- **Hiring decision**: `{rec['decision']}`")
                st.markdown(
                    f"""
                    <div class="risk-progress-container">
                        <div class="risk-progress-bar">
                            <div class="risk-progress-fill {progress_class}" 
                                 style="width: {rec['prob_risky'] * 100}%;">
                                {rec['prob_risky'] * 100:.1f}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if show_text:
                    with st.expander("Show scanned post"):
                        st.write(rec["text"])
                st.markdown("---")


if __name__ == "__main__":
    main()

