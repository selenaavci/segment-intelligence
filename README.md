# 🧠 Segment Intelligence Agent

## Project Overview  
Segment Intelligence Agent, kullanıcı tarafından yüklenen veri setleri üzerinde **otomatik segmentasyon (clustering)** yaparak, ortaya çıkan grupları **iş birimi tarafından anlaşılabilir içgörülere dönüştürmek üzere tasarlanmış** bir analiz modülüdür.

Klasik makine öğrenmesi (K-Means) ile cluster'lar oluşturulur ve her cluster için özet istatistikler, ayırt edici özellikler ve genel popülasyondan farklar çıkarılır. Sonuçlar isteğe bağlı bir LLM yorumlama katmanına gönderilmek üzere hazırlanır.

> ⚠️ **LLM entegrasyonu şu an aktif değildir.** `llm_interpreter.py` vLLM uyumlu bir HTTP çağrısı içerir ancak kod yorum satırında bırakılmıştır. Bağlantı aktifleştirilene kadar UI'da placeholder mesaj gösterilir ve cluster özeti LLM'e gönderilmeye hazır halde tutulur.

---

## 🎯 Project Purpose  
Ham veri setlerinden anlamlı segmentler çıkararak, bu segmentleri iş diliyle yorumlanabilir hale getirmek ve kurum içinde **daha hızlı, veri odaklı karar alınmasını sağlamak**.  

Teknik clustering çıktılarının ötesine geçerek, her segment için **profil tanımı, davranış analizi ve aksiyon önerileri** üretmek hedeflenir.

---

## 👥 Target Use Cases  

### 1. Customer Segmentation  
- Bireysel müşteri davranış analizi  
- Ürün sahipliği ve kanal kullanımına göre segmentasyon  
- Kampanya hedefleme ve çapraz satış fırsatları  

### 2. Commercial Segmentation  
- Ticari müşteri hacim ve işlem davranışı analizi  
- Sektör bazlı gruplaşma  
- Risk ve fırsat segmentleri  

### 3. Employee Profiling  
- Eğitim katılım ve performans analizi  
- Dijital araç kullanım segmentasyonu  
- İnsan kaynakları içgörü üretimi  

---

## ⚙️ End-to-End Workflow  

1. **Data Upload**  
   Kullanıcı CSV veya Excel (XLSX/XLS) veri setini yükler.  

2. **Automatic Data Analysis**  
   Sistem veri tiplerini otomatik olarak analiz eder:  
   - Sayısal kolonlar  
   - Kategorik kolonlar  
   - Tarih alanları  
   - ID ve anlamsız kolonlar  

3. **Feature Selection & Recommendation**  
   - Analize uygun kolonlar önerilir  
   - Gürültülü veya anlamsız feature’lar elenir  
   - Kullanıcı isterse manuel düzenleme yapabilir  

4. **Preprocessing Pipeline**  
   - Missing value handling (numeric → median, categorical → mode)  
   - Label encoding (kategorik özellikler)  
   - StandardScaler normalizasyonu  
   - Opsiyonel boyut indirgeme (PCA, kullanıcı bileşen sayısı belirler)  

5. **Clustering Execution**  
   - Algoritma: **K-Means** (tek desteklenen algoritma)  
   - Sistem silhouette score'a göre optimal cluster sayısını otomatik belirler  
   - Kullanıcı k aralığını (min/max) girer  

6. **Cluster Profiling**  
   - Her cluster için özet istatistikler çıkarılır  
   - Genel popülasyona göre fark analizi yapılır  
   - En ayırt edici feature’lar belirlenir  

7. **LLM-Based Interpretation Layer (opsiyonel, şu an placeholder)**  
   - Cluster özetleri yapılandırılmış JSON şemasına uygun şekilde hazırlanır  
   - LLM bağlandığında üretecekleri: segment adı, profil, davranış analizi, key_insights, recommended_actions, risk_notes, executive_summary, cross_segment_insights  
   - Kullanıcı opsiyonel bağlam metni (context) girebilir  

8. **Output & Reporting**  
   - Segment bazlı analiz ekranı  
   - Görselleştirmeler: silhouette grafiği, elbow grafiği, 2D cluster projeksiyonu, cluster boyut grafiği, özellik karşılaştırma, radar grafiği (3+ sayısal özellik varsa)  
   - Excel ve JSON rapor export  

9. **Feedback (UI-only)**  
   - Kullanıcı yorumların kalitesini 1–5 arasında puanlar ve yorum girebilir  
   - Not: Bu geri bildirim şu an kalıcı olarak saklanmamaktadır  

---

## 🧩 Architecture Overview  

**Core Layers:**

- **Data Processing Layer**  
  Veri temizleme, feature engineering ve preprocessing işlemleri  

- **ML Layer (Clustering Engine)**  
  K-Means tabanlı segmentasyon; silhouette + elbow ile optimal k seçimi  

- **Interpretation Layer (LLM Integration — placeholder)**  
  Cluster özetlerini LLM'e gönderilecek formata hazırlar. vLLM HTTP çağrısı kod içinde tanımlı ama yorum satırında tutulmaktadır.  

- **UI Layer (Streamlit)**  
  Kullanıcı etkileşimi, veri yükleme ve sonuç görüntüleme  

- **Export & Reporting Layer**  
  Excel / JSON çıktıları ve yönetici özetleri  

---

## 🤖 Model & Technology Stack  

### Machine Learning  
- K-Means (primary algorithm)  
- Silhouette Score (cluster validation)  
- Opsiyonel: PCA (dimension reduction)  

### LLM Integration (planned, currently placeholder)  
- vLLM uyumlu HTTP chat-completions çağrısı için hazır prompt şablonu  
- Yapılandırılmış JSON çıktı şeması tanımlı  
- Aktifleştirildiğinde: segment naming, davranış analizi, aksiyon önerileri, risk notları

### Backend & UI  
- Python  
- Pandas / NumPy  
- Scikit-learn (KMeans, PCA, StandardScaler, LabelEncoder)  
- Plotly (görselleştirmeler)  
- Streamlit  
- requests (LLM HTTP çağrısı için)  

---

## 🧠 LLM Integration Strategy  

LLM, model üretiminde değil **yorumlama katmanında** kullanılır.  

LLM’ye doğrudan ham veri değil, aşağıdaki özet bilgiler verilir:  
- Cluster bazlı istatistikler  
- Feature farkları  
- Segment büyüklüğü  
- Ayırt edici özellikler  

LLM çıktıları:  
- Segment adı  
- Profil açıklaması  
- Davranış analizi  
- Önerilen aksiyonlar  
- Risk / dikkat noktaları  

Bu yaklaşım, hem **açıklanabilirlik** hem de **kontrol edilebilirlik** sağlar.

---

## 📊 Example Output  

Her segment için sistem aşağıdaki çıktıları üretir:

- Segment Name (LLM generated)  
- Segment Size  
- Key Characteristics  
- Behavioral Profile  
- Differences from Overall Population  
- Suggested Business Actions  
- Risk Notes  

---

## 🔐 Banking & Compliance Considerations  

- Kişisel veri ve hassas veri kullanımına dikkat edilmelidir  
- Model çıktıları **karar destek aracı** olarak konumlandırılmalıdır  
- Feature selection süreci kontrol altında tutulmalıdır  
- Tüm analiz süreçleri loglanmalıdır (auditability)  
- Açıklanabilirlik (explainability) ön planda tutulmalıdır  

---

## 🚀 Business Impact  

- Veri analizi süresini ciddi ölçüde azaltır  
- Teknik olmayan ekiplerin veri içgörüsü üretmesini sağlar  
- Kampanya ve segment bazlı stratejileri hızlandırır  
- Çalışan ve müşteri davranışlarını daha iyi anlamayı sağlar  
- Kurum içinde veri odaklı karar kültürünü güçlendirir  

---


## 🔮 Future Enhancements  

- GMM / DBSCAN entegrasyonu  
- Otomatik feature importance analizi  
- Drift detection entegrasyonu  
- Campaign recommendation engine  
- Scenario simulation (what-if analysis)  
- Feedback-driven model refinement  

