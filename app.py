import streamlit as st
import torch
import json
import re

# Define the model class
class LSTMGhazalGenerator(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMGhazalGenerator, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)  # Keep Urdu letters & spaces
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("ئ", "ی").replace("ك", "ک").replace("ہ", "ہ")
    return text

# Function to generate poetry
def generate_poetry(misra1, model, vocab, idx_to_word, max_lines=5):
    device = torch.device("cpu")
    clean_misra1 = clean_text(misra1)
    input_tokens = [vocab.get(word, vocab.get("UNK", 0)) for word in clean_misra1.split()]
    poetry = [misra1]
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_lines - 1):
            output = model(input_tensor)
            predicted_tokens = torch.argmax(output, dim=2).squeeze(0).tolist()
            generated_misra = " ".join([idx_to_word.get(idx, "") for idx in predicted_tokens])
            poetry.append(generated_misra)
            input_tokens = [vocab.get(word, vocab.get("UNK", 0)) for word in generated_misra.split()]
            input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    return poetry

# Load model and vocab
@st.cache_resource
def load_model_and_vocab():
    vocab_path = "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx_to_word = {idx: word for word, idx in vocab.items()}
    vocab_size = len(vocab)
    device = torch.device("cpu")
    model = LSTMGhazalGenerator(vocab_size).to(device)
    checkpoint = torch.load("ghazal_generator.pth", map_location=device)
    model.embedding = torch.nn.Embedding.from_pretrained(checkpoint["embedding.weight"], freeze=False)
    model.fc = torch.nn.Linear(checkpoint["fc.weight"].shape[1], checkpoint["fc.weight"].shape[0])
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model, vocab, idx_to_word

# Streamlit UI
def main():
    st.set_page_config(page_title="Urdu Poetry Generator", page_icon="📜", layout="centered")
    st.sidebar.markdown("## 🌐 Language Selection")  

    language = st.sidebar.radio("Choose a language:", ["English", "اردو", "中文", "Русский"])

    
    translations = {
        "English": {"title": "🎭 Urdu Poetry Generator", "subtitle": "Enter the first verse and see the completed couplet", 
                     "input_label": "✍️ Enter the first verse", "lines_label": "📏 Number of Lines", 
                     "button_label": "📜 Generate Poetry", "warning_message": "⚠️ Please enter the first verse", 
                     "generating_message": "⏳ Generating poetry...", "result_label": "### 🌿 Generated Poetry", 
                     "about_title": "### ℹ️ About", "about_text": "This application uses an AI model to generate Urdu poetry."},
        "اردو": {"title": "🎭 اردو شاعری جنریٹر", "subtitle": "اپنا پہلا مصرع درج کریں اور مکمل شعر دیکھیں", 
                 "input_label": "✍️ پہلا مصرع درج کریں", "lines_label": "📏 مصرعوں کی تعداد", 
                 "button_label": "📜 شعر مکمل کریں", "warning_message": "⚠️ براہ کرم پہلا مصرع درج کریں", 
                 "generating_message": "⏳ شعر تیار کیا جا رہا ہے...", "result_label": "### 🌿 مکمل شعر", 
                 "about_title": "### ℹ️ بارے میں", "about_text": "یہ ایپلیکیشن مصنوعی ذہانت کے ماڈل کا استعمال کرتی ہے۔"},
        "中文": {"title": "🎭 乌尔都语诗歌生成器", "subtitle": "输入第一句诗，查看完整诗句", 
                 "input_label": "✍️ 输入第一句诗", "lines_label": "📏 诗行数", 
                 "button_label": "📜 生成诗歌", "warning_message": "⚠️ 请输入第一句诗", 
                 "generating_message": "⏳ 正在生成诗歌...", "result_label": "### 🌿 生成的诗歌", 
                 "about_title": "### ℹ️ 关于", "about_text": "此应用程序使用AI模型生成乌尔都语诗歌。"},
        "Русский": {"title": "🎭 Генератор поэзии на урду", "subtitle": "Введите первую строку и посмотрите завершённое стихотворение", 
                     "input_label": "✍️ Введите первую строку", "lines_label": "📏 Количество строк", 
                     "button_label": "📜 Сгенерировать поэзию", "warning_message": "⚠️ Пожалуйста, введите первую строку", 
                     "generating_message": "⏳ Генерация поэзии...", "result_label": "### 🌿 Сгенерированная поэзия", 
                     "about_title": "### ℹ️ О программе", "about_text": "Это приложение использует ИИ для генерации поэзии."}
    }
    
    text = translations[language]
    st.markdown(f"<p class='title'>{text['title']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtitle'>{text['subtitle']}</p>", unsafe_allow_html=True)
    
    
    try:
        model, vocab, idx_to_word = load_model_and_vocab()
        misra1 = st.text_input(text["input_label"], "", key="misra1_input")
        max_lines = st.slider(text["lines_label"], min_value=2, max_value=6, value=4)
        
        if st.button(text["button_label"]):
            if misra1:
                with st.spinner(text["generating_message"]):
                    poetry = generate_poetry(misra1, model, vocab, idx_to_word, max_lines=max_lines)
                    st.markdown(text["result_label"])
                    for misra in poetry:
                        st.markdown(f"<p class='urdu-text'>{misra}</p>", unsafe_allow_html=True)
            else:
                st.warning(text["warning_message"])
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

    st.markdown("---")
    st.markdown(text["about_title"])
    st.write(text["about_text"])

if __name__ == "__main__":
    main()
