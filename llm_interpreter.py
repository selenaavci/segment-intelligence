import json
import streamlit as st
from openai import OpenAI


def get_client():
    """Streamlit secrets'tan OpenRouter bilgilerini alarak client oluşturur."""
    api_key = st.secrets.get("LLM_API_KEY")
    base_url = st.secrets.get("LLM_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("LLM_API_KEY ve LLM_BASE_URL Streamlit Secrets'ta tanımlı değil.")
    return OpenAI(api_key=api_key, base_url=base_url)


def interpret_segments(cluster_summary, context=""):
    """LLM ile segment yorumlama yapar."""
    client = get_client()
    model = st.secrets.get("LLM_MODEL", "openrouter/free")

    system_prompt = """Sen bir veri analisti ve iş zekâsı uzmanısın.
Clustering (segmentasyon) sonuçlarını yorumlaman isteniyor.

Görevlerin:
1. Her segmente anlamlı ve akılda kalıcı bir isim ver
2. Her segment için davranışsal profil açıklaması yaz
3. Segmentler arasındaki farkları vurgula
4. Her segment için iş birimi için anlamlı içgörüler üret
5. Her segment için aksiyon önerileri sun
6. Risk veya dikkat noktalarını belirt

Çıktını aşağıdaki JSON formatında ver. Türkçe yaz.
{
  "segments": [
    {
      "id": 0,
      "name": "Segment Adı",
      "profile": "Profil açıklaması...",
      "behavioral_analysis": "Davranış analizi...",
      "key_insights": ["içgörü 1", "içgörü 2"],
      "recommended_actions": ["aksiyon 1", "aksiyon 2"],
      "risk_notes": ["risk 1"]
    }
  ],
  "executive_summary": "Yönetici özeti...",
  "cross_segment_insights": ["segmentler arası içgörü 1", "içgörü 2"]
}"""

    user_message = f"""Aşağıdaki clustering sonuçlarını yorumla:

{cluster_summary}"""

    if context:
        user_message += f"\n\nEk bağlamsal bilgi: {context}"

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    response_text = response.choices[0].message.content

    # JSON parse
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {"raw_response": response_text}
