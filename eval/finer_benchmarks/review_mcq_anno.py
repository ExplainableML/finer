from pathlib import Path
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================

IMAGE_DIR = Path("/data/rxiao/comprecap_images")

DEFAULT_CSV = "/data/rxiao/finer/eval/finer/improved_annotations/finer_comprecap/mcq_multi_obj.csv"
DEFAULT_OUT_DIR = "/data/rxiao/finer/eval/finer/improved_annotations/finer_comprecap/reviewed"

# Review pair range.
# This reviews pair 50, 51, ..., 149.
# Since every pair has 2 rows:
# pair 50 starts from rows 100 and 101.
REVIEW_START_PAIR = 50
REVIEW_END_PAIR = 150  # exclusive

REQUIRED_COLS = [
    "image",
    "qtype",
    "question",
    "choice1",
    "choice2",
    "choice3",
    "choice4",
    "choice5",
    "gt_index",
]

EDIT_TEXT_COLS = [
    "question",
    "choice1",
    "choice2",
    "choice3",
    "choice4",
    "choice5",
]


# =========================
# Helper functions
# =========================

def clear_annotation_widget_states():
    """Remove old widget states when loading a new CSV."""
    for key in list(st.session_state.keys()):
        if key.startswith("ann_"):
            del st.session_state[key]


def load_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV does not exist: {path}")

    df = pd.read_csv(path, keep_default_na=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only required columns, in fixed order.
    df = df[REQUIRED_COLS].copy()

    # Normalize gt_index.
    df["gt_index"] = pd.to_numeric(df["gt_index"], errors="coerce").fillna(0).astype(int)

    return df


def role_from_qtype(qtype: str, fallback: str) -> str:
    qtype = str(qtype)
    if qtype.endswith("_pos"):
        return "Positive"
    if qtype.endswith("_neg"):
        return "Negative"
    return fallback


def gt_choice_text(row: pd.Series) -> str:
    gt_idx = int(row["gt_index"])
    if 0 <= gt_idx <= 4:
        return str(row[f"choice{gt_idx + 1}"])
    return "INVALID_GT_INDEX"


def render_editable_row(df: pd.DataFrame, row_idx: int, role: str):
    row = df.loc[row_idx]

    st.markdown(f"### {role} row")
    st.markdown(f"**Row index:** `{row_idx}`")
    st.markdown(f"**qtype:** `{row['qtype']}`")

    for col in EDIT_TEXT_COLS:
        key = f"ann_{row_idx}_{col}"
        value = st.text_area(
            label=col,
            value=str(row[col]),
            key=key,
            height=90 if col == "question" else 70,
        )
        df.at[row_idx, col] = value

    current_gt = int(df.at[row_idx, "gt_index"])
    if current_gt < 0 or current_gt > 4:
        current_gt = 0

    gt_key = f"ann_{row_idx}_gt_index"
    new_gt = st.selectbox(
        "gt_index",
        options=[0, 1, 2, 3, 4],
        index=current_gt,
        format_func=lambda x: f"{x}  →  choice{x + 1}",
        key=gt_key,
    )
    df.at[row_idx, "gt_index"] = int(new_gt)

    st.success(f"GT choice: choice{new_gt + 1}")
    st.write(df.at[row_idx, f"choice{new_gt + 1}"])


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="MCQ Annotation Reviewer",
    layout="wide",
)

st.title("MCQ Annotation Reviewer")

with st.sidebar:
    st.header("Input / Output")

    csv_path = st.text_input("Input CSV path", value=DEFAULT_CSV)
    out_dir = st.text_input("Output directory", value=DEFAULT_OUT_DIR)

    st.markdown("---")
    st.header("Review Range")

    review_start_pair_input = st.number_input(
        "Start pair index",
        min_value=0,
        value=REVIEW_START_PAIR,
        step=1,
    )

    review_end_pair_input = st.number_input(
        "End pair index, exclusive",
        min_value=1,
        value=REVIEW_END_PAIR,
        step=1,
    )

    load_button = st.button("Load CSV", type="primary")

    st.markdown("---")
    st.write("Image directory:")
    st.code(str(IMAGE_DIR))

    st.write(
        f"Configured to show pair `{review_start_pair_input}` "
        f"to pair `{review_end_pair_input - 1}`."
    )


if load_button:
    try:
        df = load_csv(csv_path)
        st.session_state["df"] = df
        st.session_state["csv_path"] = csv_path
        clear_annotation_widget_states()
        st.success(f"Loaded CSV: {csv_path}")
    except Exception as e:
        st.error(str(e))


if "df" not in st.session_state:
    st.info("Enter a CSV path and click **Load CSV**.")
    st.stop()


df = st.session_state["df"]
csv_path_loaded = Path(st.session_state["csv_path"])

num_pairs_total = len(df) // 2

review_start_pair = int(review_start_pair_input)
review_end_pair = int(review_end_pair_input)

review_start_pair = max(0, review_start_pair)
review_end_pair = min(review_end_pair, num_pairs_total)

if review_start_pair >= review_end_pair:
    st.error(
        f"Invalid review range: pair {review_start_pair} to pair {review_end_pair}. "
        f"Total pairs: {num_pairs_total}"
    )
    st.stop()

num_pairs_to_show = review_end_pair - review_start_pair

st.write(f"Loaded file: `{csv_path_loaded}`")
st.write(f"Total rows: `{len(df)}`")
st.write(f"Total pairs: `{num_pairs_total}`")
st.write(f"Reviewing pair range: `{review_start_pair}` to `{review_end_pair - 1}`")
st.write(f"Currently showing: `{num_pairs_to_show}` pairs")

if len(df) % 2 != 0:
    st.warning("The CSV has an odd number of rows. The last row will not be shown as a pair.")


# =========================
# Save button
# =========================

top_save_col, _ = st.columns([1, 4])

with top_save_col:
    if st.button("Save corrected CSV", type="primary"):
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / csv_path_loaded.name
        df.to_csv(out_path, index=False)

        st.success(f"Saved to: {out_path}")


# =========================
# Pair visualization
# =========================

for pair_id in range(review_start_pair, review_end_pair):
    row_a = pair_id * 2
    row_b = pair_id * 2 + 1

    qtype_a = str(df.loc[row_a, "qtype"])
    qtype_b = str(df.loc[row_b, "qtype"])

    role_a = role_from_qtype(qtype_a, "Row A")
    role_b = role_from_qtype(qtype_b, "Row B")

    # Prefer positive on the left and negative on the right.
    if role_a == "Negative" and role_b == "Positive":
        left_idx, right_idx = row_b, row_a
        left_role, right_role = role_b, role_a
    else:
        left_idx, right_idx = row_a, row_b
        left_role, right_role = role_a, role_b

    image_name = str(df.loc[left_idx, "image"])
    image_path = IMAGE_DIR / image_name

    with st.expander(
        f"Pair {pair_id:03d} | rows {row_a}, {row_b} | image: {image_name}",
        expanded=(pair_id == review_start_pair),
    ):
        img_col, edit_col = st.columns([1, 2])

        with img_col:
            st.subheader("Image")
            if image_path.exists():
                st.image(str(image_path), caption=image_name, use_container_width=True)
            else:
                st.error(f"Image not found: {image_path}")

            st.markdown("#### Current GT summary")
            for idx, role in [(left_idx, left_role), (right_idx, right_role)]:
                row = df.loc[idx]
                gt_idx = int(row["gt_index"])
                st.markdown(f"**{role}** | row `{idx}` | gt_index `{gt_idx}`")
                st.write(gt_choice_text(row))

        with edit_col:
            col1, col2 = st.columns(2)

            with col1:
                render_editable_row(df, left_idx, left_role)

            with col2:
                render_editable_row(df, right_idx, right_role)