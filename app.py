# app.py
import uuid
import streamlit as st
from langgraph.types import Command
from graph import get_app
from agents.topic_validator import validate_topic

# ── Singleton app instance ────────────────────────────────────────────────────
app = get_app()

st.set_page_config(
    page_title="5-Module AI Mastery Tutor",
    page_icon="🎓",
    layout="wide"
)

# ── Session state initialisation ──────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"tutor_session_{uuid.uuid4().hex[:8]}"
if "course_started" not in st.session_state:
    st.session_state.course_started = False
if "topic_error" not in st.session_state:
    st.session_state.topic_error = ""
if "validated_topic" not in st.session_state:
    st.session_state.validated_topic = ""

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("🎓 5-Step Adaptive AI Tutor & Capstone Evaluator")

# ── Sidebar ───────────────────────────────────────────────────────────────────
topic_input = st.sidebar.text_input(
    "Enter Topic to Master:",
    "Transformer Architectures & Attention Mechanisms"
)

if st.sidebar.button("🔄 Start New Course"):
    st.session_state.thread_id = f"tutor_session_{uuid.uuid4().hex[:8]}"
    st.session_state.course_started = False
    st.session_state.topic_error = ""
    st.session_state.validated_topic = ""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.rerun()

if st.sidebar.button("▶️ Start 5-Step Course"):
    if st.session_state.course_started:
        st.sidebar.warning("Course already running. Click '🔄 Start New Course' to restart.")
    else:
        # ── Topic validation ──────────────────────────────────────────────
        with st.spinner("🔍 Validating topic..."):
            validation = validate_topic(topic_input)

        if not validation["valid"]:
            st.session_state.topic_error = validation["reason"]
            st.rerun()
        else:
            final_topic = validation["cleaned"] or topic_input.strip()
            st.session_state.topic_error = ""
            st.session_state.validated_topic = final_topic

            with st.spinner(f"📚 Building your personalised curriculum for: **{final_topic}**..."):
                app.invoke({"main_topic": final_topic}, config)
            st.session_state.course_started = True
            st.rerun()

# ── Sidebar status display ────────────────────────────────────────────────────
if st.session_state.topic_error:
    st.sidebar.error(
        f"❌ Invalid topic: {st.session_state.topic_error}\n\n"
        f"Please enter a specific subject, technology, or skill."
    )

if st.session_state.validated_topic and st.session_state.course_started:
    st.sidebar.success(f"📖 Active topic:\n**{st.session_state.validated_topic}**")

# ── Guard: nothing to show until course starts ────────────────────────────────
if not st.session_state.course_started:
    st.info("👈 Enter a topic in the sidebar and click **▶️ Start 5-Step Course** to begin.")
    st.stop()

# ── Fetch current graph state ─────────────────────────────────────────────────
snapshot = app.get_state(config)

# ── Interrupt handling ────────────────────────────────────────────────────────
if snapshot.next and snapshot.tasks and snapshot.tasks[0].interrupts:
    payload = snapshot.tasks[0].interrupts[0].value

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE QUIZ
    # ══════════════════════════════════════════════════════════════════════════
    if payload.get("type") == "module_quiz":

        step        = payload.get("step", "?")
        title       = payload.get("title", "Current Module")
        explanation = payload.get("explanation", "No explanation available.")
        yt_links    = payload.get("youtube_links", [])
        mcqs        = payload.get("mcqs", [])   # sanitised — no correct_index

        st.header(f"📘 Module {step}/5: {title}")

        # ── Progress bar ──────────────────────────────────────────────────
        try:
            progress = int(step) / 5
        except (ValueError, TypeError):
            progress = 0.0
        st.progress(progress, text=f"Module {step} of 5")

        # ── Lesson content ────────────────────────────────────────────────
        st.subheader("📖 Lesson")
        st.markdown(explanation)

        # ── YouTube resources ─────────────────────────────────────────────
        if yt_links:
            st.subheader("📺 Deep-Dive YouTube Resources")
            for yt in yt_links:
                q_text = yt.get("query", "Resource")
                url    = yt.get("url", "#")
                st.markdown(f"- [{q_text}]({url})")

        st.divider()

        # ── Previous module score (shown before new quiz) ─────────────────
        module_scores = snapshot.values.get("module_scores", [])
        if module_scores:
            last       = module_scores[-1]
            last_step  = last.get("step", "?")
            last_title = last.get("title", "")
            score_pct  = last.get("score", 0)
            correct    = last.get("correct", "?")
            total      = last.get("total", "?")

            st.subheader(f"📊 Module {last_step} Results: {last_title}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Score",   f"{score_pct}%")
            col2.metric("Correct", f"{correct}/{total}")
            col3.metric("Status",  "✅ Passed" if score_pct >= 50 else "❌ Needs Review")

            if score_pct == 100:
                st.success("🎯 Perfect score on the previous module!")
            elif score_pct >= 50:
                st.warning(f"⚠️ You got {correct}/{total} correct on Module {last_step}.")
            else:
                st.error(
                    f"❌ Only {correct}/{total} correct on Module {last_step}. "
                    f"Review the material above before proceeding."
                )

            # ── Per-question answer review ────────────────────────────────
            # Full MCQ data (with correct_index) lives in current_module_data
            # but that is now the NEW module. Previous module answers are
            # in student_mcq_answers and we need previous module's MCQ data.
            # We store it via module_scores which we enriched with correct answers.
            prev_answers = snapshot.values.get("student_mcq_answers", {})
            prev_mcq_detail = last.get("mcq_detail", [])

            if prev_answers and prev_mcq_detail:
                with st.expander("🔍 See detailed answer breakdown for previous module"):
                    for q in prev_mcq_detail:
                        q_id         = str(q.get("id", ""))
                        question     = q.get("question", "")
                        options      = q.get("options", [])
                        correct_idx  = q.get("correct_index", 0)
                        expl         = q.get("explanation", "")
                        correct_text = options[correct_idx] if options and correct_idx < len(options) else ""
                        selected     = prev_answers.get(q_id, "")

                        st.write(f"**Q: {question}**")
                        if selected == correct_text:
                            st.success(f"✅ Your answer: **{selected}** — Correct!")
                        else:
                            st.error(f"❌ Your answer: **{selected}**")
                            st.info(f"✔️ Correct answer: **{correct_text}**")
                        if expl:
                            st.caption(f"💡 Explanation: {expl}")
                        st.write("")

            st.divider()

        # ── Knowledge check form ──────────────────────────────────────────
        st.subheader("✍️ Module Knowledge Check")
        st.caption("Answer all questions below to proceed to the next module.")

        with st.form("quiz_form"):
            user_ans     = {}
            all_answered = True

            for q in mcqs:
                if not isinstance(q, dict):
                    continue

                question_text = q.get("question", "No question text.")
                options       = q.get("options", [])
                key_str       = str(q.get("id", "unknown"))

                st.write(f"**{question_text}**")

                if not options:
                    st.warning(f"Question {key_str} has no options — skipping.")
                    continue

                selected = st.radio(
                    "Select your answer:",
                    options,
                    index=None,         # force explicit selection, no default
                    key=f"q_{key_str}"
                )

                if selected is None:
                    all_answered = False
                else:
                    user_ans[key_str] = selected   # stores full option text

            submitted = st.form_submit_button("✅ Submit Answers & Continue")

            if submitted:
                if not all_answered:
                    st.warning("⚠️ Please answer ALL questions before submitting.")
                else:
                    with st.spinner("⏳ Saving answers and loading next module..."):
                        app.invoke(Command(resume=user_ans), config)
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # CAPSTONE SUBMISSION
    # ══════════════════════════════════════════════════════════════════════════
    elif payload.get("type") == "capstone_submission":
        st.header("🏆 Final Capstone Project")
        st.progress(1.0, text="All 5 modules complete — Final project")

        project_prompt = payload.get(
            "project_prompt",
            "Complete your final capstone project."
        )
        st.info(project_prompt)

        # ── Show full course score summary before submission ──────────────
        module_scores = snapshot.values.get("module_scores", [])
        if module_scores:
            st.subheader("📊 Your Course Performance Summary")
            cols = st.columns(len(module_scores))
            for i, s in enumerate(module_scores):
                score_pct = s.get("score", 0)
                cols[i].metric(
                    label=f"Module {s.get('step','?')}",
                    value=f"{score_pct}%",
                    delta=f"{s.get('correct','?')}/{s.get('total','?')} correct"
                )
            overall = sum(s.get("score", 0) for s in module_scores) / len(module_scores)
            st.metric("📈 Overall Course Average", f"{overall:.1f}%")
            st.divider()

        with st.form("project_form"):
            submission = st.text_area(
                "Paste your project code or written report here:",
                height=350,
                placeholder="Minimum 50 characters required. Include your implementation, "
                            "reasoning, and any results or observations."
            )

            submitted = st.form_submit_button("🚀 Submit Capstone for Final Evaluation")

            if submitted:
                if not submission or len(submission.strip()) < 50:
                    st.warning(
                        "⚠️ Submission too short. "
                        "Please provide a meaningful response (minimum 50 characters)."
                    )
                else:
                    with st.spinner("🔍 Evaluating your capstone project..."):
                        app.invoke(Command(resume=submission.strip()), config)
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# COURSE COMPLETION
# ══════════════════════════════════════════════════════════════════════════════
elif snapshot.values and snapshot.values.get("is_mastery_completed"):
    st.balloons()
    st.success("🎉 CONGRATULATIONS! You have completed the course!")
    st.progress(1.0, text="Course Complete ✅")

    # ── Final module score summary ────────────────────────────────────────
    module_scores = snapshot.values.get("module_scores", [])
    if module_scores:
        st.subheader("📊 Full Course Performance")
        cols = st.columns(len(module_scores))
        for i, s in enumerate(module_scores):
            score_pct = s.get("score", 0)
            cols[i].metric(
                label=f"Module {s.get('step','?')}",
                value=f"{score_pct}%",
                delta=f"{s.get('correct','?')}/{s.get('total','?')} correct"
            )
        overall = sum(s.get("score", 0) for s in module_scores) / len(module_scores)
        st.metric("📈 Final Course Average", f"{overall:.1f}%")
        st.divider()

    # ── Project evaluation report ─────────────────────────────────────────
    st.subheader("📋 Capstone Evaluation Report")
    feedback = snapshot.values.get(
        "project_feedback",
        "Your project has been evaluated. Well done on completing the course!"
    )
    st.markdown(feedback)

    st.divider()
    if st.button("🔄 Start a New Course"):
        st.session_state.thread_id = f"tutor_session_{uuid.uuid4().hex[:8]}"
        st.session_state.course_started = False
        st.session_state.topic_error = ""
        st.session_state.validated_topic = ""
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING STATE (graph running, no interrupt yet)
# ══════════════════════════════════════════════════════════════════════════════
else:
    if st.session_state.course_started:
        st.info("⏳ Processing your course content... Please wait.")
        with st.spinner("Loading next module..."):
            st.empty()