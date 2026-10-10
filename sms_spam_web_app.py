import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torch.nn.utils.rnn import pack_padded_sequence

# =====================================================================
# 1. Page config (must be the first Streamlit command)
# =====================================================================
st.set_page_config(
    page_title="SMS Spam Classification",
    page_icon="✉️",
    layout="wide",
)

MODEL_PATH = "bilstm_spam.pth"
MAX_CHARS = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 2. Model definition + loading + prediction
# =====================================================================
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        packed_embedded = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed_embedded)

        # hidden[0] = forward final state, hidden[1] = backward final state
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        return self.fc(hidden)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the checkpoint once and keep it in memory between reruns."""
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    vocab = checkpoint["vocab"]
    model = BiLSTMModel(
        checkpoint["vocab_size"],
        checkpoint["embedding_dim"],
        checkpoint["hidden_dim"],
        checkpoint["output_dim"],
        checkpoint["embedding_matrix"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def predict_spam(model, vocab, message):
    """Return (not_spam_probability, spam_probability) or None for empty text."""
    words = message.split()
    if not words:
        return None

    unk = vocab["<UNK>"]
    indices = [vocab.get(word, unk) for word in words]

    x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(indices)], dtype=torch.long)

    with torch.no_grad():
        probs = F.softmax(model(x, lengths), dim=1)[0]

    return probs[0].item(), probs[1].item()


# =====================================================================
# 3. Styling
# =====================================================================
st.markdown("""
<style>
    /* ===== App background + decorative circles ===== */
    .stApp {
        background:
            radial-gradient(ellipse at 5% 5%, #E8EDFF 0%, transparent 28%),
            radial-gradient(ellipse at 95% 85%, #EEF0FF 0%, transparent 25%),
            linear-gradient(180deg, #F8FAFF 0%, #F2F6FD 100%);
    }

    .stApp::before,
    .stApp::after,
    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        border-radius: 50%;
        background: #E6EBFF;
        opacity: 0.55;
        pointer-events: none;
        z-index: 0;
    }
    .stApp::before { width: 240px; height: 240px; top: 20px; left: -130px; }
    .stApp::after { width: 220px; height: 220px; top: 60px; right: -110px; }
    [data-testid="stAppViewContainer"]::before { width: 260px; height: 260px; bottom: -120px; left: -120px; }
    [data-testid="stAppViewContainer"]::after { width: 260px; height: 260px; bottom: -110px; right: -110px; }

    /* ===== Hide default Streamlit chrome ===== */
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        display: none !important;
    }

    /* ===== Fixed top navigation bar ===== */
    .top-navbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        width: 100%;
        height: 66px;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 4%;
        background: #202D4B;
        border-bottom: 1px solid #34415F;
        box-shadow: 0 4px 18px rgba(30, 45, 80, 0.10);
        z-index: 999999;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #FFFFFF !important;
        font-size: 1.2rem;
        font-weight: 750;
        white-space: nowrap;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        flex-shrink: 0;
        border-radius: 13px;
        background: linear-gradient(135deg, #7768FF, #5145E8);
        color: #FFFFFF !important;
        font-size: 1.5rem;
        box-shadow: 0 4px 12px rgba(101, 85, 245, 0.25);
    }

    .nav-caption { color: #D5DCEE !important; font-size: 0.85rem; white-space: nowrap; }
    .heart { color: #FF6685 !important; }

    /* ===== Content area ===== */
    .block-container {
        position: relative;
        z-index: 1;
        max-width: 1240px !important;
        padding-top: 6rem !important;
        padding-bottom: 2rem !important;
    }

    /* ===== Hero ===== */
    .hero {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 22px;
        padding-top: 0.5rem;
    }

    .hero-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 84px;
        height: 84px;
        border-radius: 50%;
        background: #E4E9FF;
        font-size: 2.2rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #1F2A4D;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #66789A !important;
        font-size: 1.35rem;
        margin: 0.6rem 0 0.8rem;
    }

    .powered {
        text-align: center;
        color: #7A89A8 !important;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }

    /* ===== Cards ===== */
    .section-card,
    .st-key-input_card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem 1.5rem 1.2rem;
        border-radius: 20px;
        border: 1px solid #E7ECF7;
        box-shadow: 0 8px 28px rgba(55, 84, 170, 0.06);
        margin-bottom: 1.2rem;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        color: #202F53 !important;
        margin-bottom: 0.9rem;
    }

    .section-title.plain { margin: 0.4rem 0 0.8rem; padding-left: 0.3rem; }

    /* ===== Text area ===== */
    div[data-testid="stTextArea"] textarea {
        background-color: #FFFFFF;
        color: #263657;
        border: 1px solid #DCE4F2;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        line-height: 1.6;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #665AF5;
        box-shadow: 0 0 0 3px rgba(102, 90, 245, 0.10);
    }

    div[data-testid="stTextArea"] textarea::placeholder { color: #8A98B5; }
    [data-testid="InputInstructions"] { color: #8A98B5 !important; }

    /* ===== "Check Message" button ===== */
    .st-key-check_btn button {
        width: 100%;
        background: linear-gradient(100deg, #4F46E5, #4361EE);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        box-shadow: 0 5px 14px rgba(79, 70, 229, 0.25);
        transition: all 0.2s ease;
    }

    .st-key-check_btn button,
    .st-key-check_btn button p {
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 1rem;
    }

    .st-key-check_btn button:hover {
        background: linear-gradient(100deg, #4338CA, #3555E0);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.35);
        transform: translateY(-2px);
        border: none;
    }

    /* ===== Example cards (they are real buttons) ===== */
    [class*="st-key-ex_"] button {
        width: 100%;
        min-height: 96px;
        height: auto;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 12px;
        text-align: left;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: #F4F6FF;
        border: 1px solid #E6EAF8;
        box-shadow: none;
        transition: all 0.2s ease;
    }

    [class*="st-key-ex_"] button p {
        color: #3A4A86 !important;
        font-size: 0.9rem;
        font-weight: 500;
        line-height: 1.5;
        text-align: left;
        white-space: normal;
    }

    [class*="st-key-ex_spam"] button:hover {
        border-color: #F5B5BE;
        background: #FFF3F5;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(229, 72, 77, 0.10);
    }

    [class*="st-key-ex_ham"] button:hover {
        border-color: #A9E3C2;
        background: #F0FBF5;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(18, 161, 80, 0.10);
    }

    /* ===== Result: empty state ===== */
    .empty-state { text-align: center; padding: 1.2rem 0 1.6rem; }

    .empty-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: #E9EDFA;
        font-size: 1.6rem;
        margin-bottom: 0.9rem;
        opacity: 0.8;
    }

    .empty-title { font-size: 1.1rem; font-weight: 700; color: #5A6B8F; }
    .empty-sub { font-size: 0.9rem; color: #7A89A8; margin-top: 0.4rem; }

    /* ===== Result: filled state ===== */
    .result-body { text-align: center; padding: 0.6rem 0 0.4rem; }

    .result-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 72px;
        height: 72px;
        border-radius: 50%;
        font-size: 2rem;
        margin-bottom: 0.6rem;
    }

    .result-label { font-size: 2.4rem; font-weight: 800; line-height: 1.2; }
    .result-desc { color: #66789A; margin: 0.3rem 0 1.4rem; }

    .result-body.spam .result-icon { background: #FFE9EC; }
    .result-body.spam .result-label { color: #E5484D; }
    .result-body.ham .result-icon { background: #E3F7EC; }
    .result-body.ham .result-label { color: #12A150; }

    .prob-row {
        display: flex;
        align-items: center;
        gap: 14px;
        max-width: 560px;
        margin: 0.6rem auto;
        font-size: 0.92rem;
        color: #435474;
    }

    .prob-name { width: 80px; text-align: left; font-weight: 600; }
    .prob-pct { width: 64px; text-align: right; font-weight: 700; }
    .bar { flex: 1; height: 10px; border-radius: 99px; background: #E9EDF7; overflow: hidden; }
    .fill { height: 100%; border-radius: 99px; }
    .fill.ham { background: linear-gradient(90deg, #34C77B, #12A150); }
    .fill.spam { background: linear-gradient(90deg, #FF7A85, #E5484D); }

    /* ===== Responsive ===== */
    @media (max-width: 768px) {
        .top-navbar { height: 60px; padding: 0 14px; }
        .brand { gap: 8px; font-size: 0.95rem; }
        .brand-icon { width: 36px; height: 36px; font-size: 1.25rem; }
        .nav-caption { font-size: 0.7rem; }
        .block-container { padding-top: 5.5rem !important; }
        .hero { gap: 12px; }
        .hero-icon { width: 54px; height: 54px; font-size: 1.5rem; }
        .hero-title { font-size: 1.7rem; }
        .subtitle { font-size: 1rem; padding: 0 0.5rem; }
        .section-card, .st-key-input_card { padding: 1.1rem; border-radius: 16px; }
        .result-label { font-size: 1.9rem; }
        .prob-name { width: 66px; }
    }

    @media (max-width: 420px) {
        .nav-caption { display: none; }
    }
</style>
""", unsafe_allow_html=True)


def render(markup: str):
    """Render HTML safely: strip indentation/newlines so Markdown never treats it as a code block."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


# =====================================================================
# 4. Static layout: navbar + hero
# =====================================================================
render("""
<div class="top-navbar">
    <div class="brand">
        <div class="brand-icon">✉</div>
        <span style="color:#FFFFFF;">SMS Spam Detector</span>
    </div>
    <div class="nav-caption">
        <span>Built with <span class="heart">♥</span> PyTorch &amp; Streamlit</span>
    </div>
</div>
""")

render("""
<div class="hero">
    <div class="hero-icon">💬</div>
    <div class="hero-title">SMS Spam Classification</div>
</div>
<div class="subtitle">Paste your message below and let the model decide if it’s spam or not.</div>
<div class="powered">Powered by a BiLSTM model with GloVe embeddings</div>
""")


# =====================================================================
# 5. State + callbacks
# =====================================================================
EXAMPLES = [
    ("spam", "🚨", "Congratulations! You have won a free prize. Call now to claim your reward!"),
    ("ham", "💬", "Hey, are we still meeting tomorrow? Let me know what time works for you."),
    ("spam", "🚨", "You've been selected for a limited time offer! Click here now!"),
    ("ham", "💬", "Sounds good! See you at 7pm."),
]

if "msg" not in st.session_state:
    st.session_state.msg = ""
if "result" not in st.session_state:
    st.session_state.result = None


def use_example(text: str):
    st.session_state.msg = text
    st.session_state.result = None


# =====================================================================
# 6. Input card
# =====================================================================
model, vocab = load_model()

with st.container(key="input_card"):
    render('<div class="section-title">✏️ Enter your message</div>')

    st.text_area(
        "Message",
        key="msg",
        height=120,
        max_chars=MAX_CHARS,
        placeholder="Type your SMS message here...",
        label_visibility="collapsed",
    )

    _, btn_col = st.columns([3, 1])
    with btn_col:
        check = st.button("Check Message", key="check_btn", icon="📨")

    if check:
        text = st.session_state.msg.strip()
        if not text:
            st.session_state.result = None
            st.warning("Please enter a message first.")
        else:
            st.session_state.result = predict_spam(model, vocab, text)


# =====================================================================
# 7. Example cards
# =====================================================================
render('<div class="section-title plain">💡 Try these examples</div>')

cols = st.columns(len(EXAMPLES))
for i, (col, (kind, icon, text)) in enumerate(zip(cols, EXAMPLES)):
    with col:
        st.button(
            text,
            key=f"ex_{kind}_{i}",
            icon=icon,
            on_click=use_example,
            args=(text,),
        )


# =====================================================================
# 8. Prediction result card
# =====================================================================
result = st.session_state.result

if result is None:
    render("""
    <div class="section-card">
        <div class="section-title">📊 Prediction Result</div>
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div class="empty-title">Your prediction will appear here</div>
            <div class="empty-sub">Enter a message and click “Check Message”</div>
        </div>
    </div>
    """)
else:
    p_ham, p_spam = result
    is_spam = p_spam > p_ham
    kind = "spam" if is_spam else "ham"
    icon = "🚨" if is_spam else "✅"
    label = "Spam" if is_spam else "Not Spam"
    desc = (
        f"The model is {max(p_ham, p_spam) * 100:.1f}% confident this message is "
        f"{'spam' if is_spam else 'legitimate'}."
    )

    render(f"""
    <div class="section-card">
        <div class="section-title">📊 Prediction Result</div>
        <div class="result-body {kind}">
            <div class="result-icon">{icon}</div>
            <div class="result-label">{label}</div>
            <div class="result-desc">{desc}</div>
            <div class="prob-row">
                <span class="prob-name">Not Spam</span>
                <div class="bar"><div class="fill ham" style="width:{p_ham * 100:.1f}%"></div></div>
                <span class="prob-pct">{p_ham * 100:.2f}%</span>
            </div>
            <div class="prob-row">
                <span class="prob-name">Spam</span>
                <div class="bar"><div class="fill spam" style="width:{p_spam * 100:.1f}%"></div></div>
                <span class="prob-pct">{p_spam * 100:.2f}%</span>
            </div>
        </div>
    </div>
    """)
