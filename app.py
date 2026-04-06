import streamlit as st
import pandas as pd
import numpy as np

from data_processor import load_data, analyze_columns, recommend_features, preprocess_data
from clustering_engine import find_optimal_k, run_clustering, generate_cluster_profiles, prepare_llm_summary
from llm_interpreter import interpret_segments
from visualizer import (
    plot_silhouette_scores,
    plot_elbow,
    plot_2d_clusters,
    plot_cluster_sizes,
    plot_feature_comparison,
    plot_radar_chart,
)
from exporter import export_to_excel, export_to_json

# --- Sayfa ayarları ---
st.set_page_config(
    page_title="Segment Intelligence Agent",
    page_icon="🧠",
    layout="wide",
)

st.title("Segment Intelligence Agent")
st.caption("Verinizi yükleyin, segmentleri keşfedelim, yapay zekâ ile yorumlayalım.")

# --- Session state başlangıç ---
for key in ["df", "analysis", "selected_features", "labels", "profiles", "llm_result", "cluster_results_df", "scaled_df"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =====================================================
# ADIM 1: Veri Yükleme
# =====================================================
st.header("1. Veri Yükleme")

uploaded_file = st.file_uploader("CSV veya Excel dosyanızı yükleyin", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success(f"Veri başarıyla yüklendi: {df.shape[0]} satır, {df.shape[1]} kolon")

        with st.expander("Veri Önizleme", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Veri Tipleri:**")
                st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Kolon", 0: "Tip"}))
            with col2:
                st.write("**Eksik Değer Özeti:**")
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) > 0:
                    st.dataframe(missing.reset_index().rename(columns={"index": "Kolon", 0: "Eksik Sayı"}))
                else:
                    st.info("Eksik değer yok.")
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")

# =====================================================
# ADIM 2: Otomatik Kolon Analizi & Feature Seçimi
# =====================================================
if st.session_state.df is not None:
    st.header("2. Özellik Seçimi")

    df = st.session_state.df
    analysis = analyze_columns(df)
    st.session_state.analysis = analysis

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sayısal Kolonlar:**", ", ".join(analysis["numeric"]) if analysis["numeric"] else "Yok")
        st.write("**Kategorik Kolonlar:**", ", ".join(analysis["categorical"]) if analysis["categorical"] else "Yok")
    with col2:
        st.write("**Tarih Alanları:**", ", ".join(analysis["datetime"]) if analysis["datetime"] else "Yok")
        st.write("**Elenen Kolonlar (ID/Anlamsız):**", ", ".join(analysis["id_or_useless"]) if analysis["id_or_useless"] else "Yok")

    recommended, excluded = recommend_features(analysis)

    if not recommended:
        st.warning("Analiz için uygun kolon bulunamadı. Lütfen verinizdeki kolonları kontrol edin.")
        st.stop()

    selected = st.multiselect(
        "Analize dahil edilecek özellikleri seçin:",
        options=list(df.columns),
        default=recommended,
        help="Sistem otomatik olarak uygun kolonları önermiştir. İsterseniz düzenleyebilirsiniz.",
    )

    st.session_state.selected_features = selected

    # =====================================================
    # ADIM 3: Ön İşleme Ayarları
    # =====================================================
    if selected:
        st.header("3. Ön İşleme ve Kümeleme")

        col1, col2 = st.columns(2)
        with col1:
            apply_pca = st.checkbox("PCA (Boyut İndirgeme) Uygula", value=False)
            pca_components = 2
            if apply_pca:
                max_comp = min(len(selected), 10)
                pca_components = st.slider("PCA Bileşen Sayısı", 2, max_comp, 2)

        with col2:
            k_min = st.number_input("Minimum Küme Sayısı", min_value=2, max_value=20, value=2)
            k_max = st.number_input("Maksimum Küme Sayısı", min_value=2, max_value=20, value=min(10, len(df) - 1))

        # =====================================================
        # ADIM 4: Kümeleme Çalıştırma
        # =====================================================
        if st.button("Segmentasyonu Başlat", type="primary", use_container_width=True):
            with st.spinner("Veri işleniyor..."):
                clustering_data, scaled_df, scaler, encoders, pca_model = preprocess_data(
                    df, selected, apply_pca, pca_components
                )
                st.session_state.scaled_df = scaled_df

            with st.spinner("Optimal küme sayısı hesaplanıyor..."):
                best_k, results_df = find_optimal_k(clustering_data, k_range=(k_min, k_max))
                st.session_state.cluster_results_df = results_df

            st.subheader("Optimal Küme Analizi")
            tab1, tab2 = st.tabs(["Silhouette Skoru", "Dirsek Yöntemi"])
            with tab1:
                st.plotly_chart(plot_silhouette_scores(results_df), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_elbow(results_df), use_container_width=True)

            # Kullanıcı geçersiz kılma
            chosen_k = st.number_input(
                f"Küme sayısı (önerilen: {best_k}):",
                min_value=2,
                max_value=int(k_max),
                value=int(best_k),
                key="chosen_k",
            )

            with st.spinner("Kümeleme çalışıyor..."):
                labels, kmeans_model, sil_score = run_clustering(clustering_data, chosen_k)
                st.session_state.labels = labels

            st.success(f"Kümeleme tamamlandı! Silhouette Skoru: {sil_score:.3f}")

            # =====================================================
            # ADIM 5: Segment Profilleme
            # =====================================================
            with st.spinner("Segment profilleri oluşturuluyor..."):
                profiles = generate_cluster_profiles(df, selected, labels)
                st.session_state.profiles = profiles

            st.header("4. Segment Analizi")

            # Segment büyüklükleri
            st.plotly_chart(plot_cluster_sizes(profiles), use_container_width=True)

            # 2D Scatter
            st.plotly_chart(plot_2d_clusters(clustering_data, labels, selected), use_container_width=True)

            # Radar chart
            numeric_feats = [f for f in selected if pd.api.types.is_numeric_dtype(df[f])]
            if len(numeric_feats) >= 3:
                radar = plot_radar_chart(profiles, numeric_feats)
                if radar:
                    st.plotly_chart(radar, use_container_width=True)

            # Özellik karşılaştırma
            st.subheader("Özellik Karşılaştırması")
            if numeric_feats:
                feat_to_compare = st.selectbox("Özellik seçin:", numeric_feats)
                fig = plot_feature_comparison(profiles, feat_to_compare)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # Detaylı profiller
            with st.expander("Detaylı Segment Profilleri"):
                for cid, p in profiles.items():
                    st.markdown(f"**Segment {cid}** — {p['size']} kayıt (%{p['percentage']})")
                    if p["distinguishing_features"]:
                        for feat in p["distinguishing_features"]:
                            st.markdown(f"- {feat}")
                    st.divider()

            # =====================================================
            # ADIM 6: LLM Yorumlama
            # =====================================================
            st.header("5. Yapay Zekâ Yorumlama")

            context = st.text_area(
                "Ek bağlamsal bilgi (isteğe bağlı):",
                placeholder="Örneğin: Bu veri seti banka bireysel müşterilerine aittir. Amacımız kampanya hedeflemesi yapmaktır.",
                help="Yapay zekânın daha isabetli yorumlar üretmesi için veri seti hakkında ek bilgi verebilirsiniz.",
            )

            if st.button("Yapay Zekâ ile Yorumla", type="primary", use_container_width=True):
                try:
                    llm_summary = prepare_llm_summary(profiles, selected)

                    with st.spinner("Yapay zekâ segmentleri yorumluyor..."):
                        llm_result = interpret_segments(llm_summary, context)
                        st.session_state.llm_result = llm_result

                    if "raw_response" in llm_result:
                        st.markdown(llm_result["raw_response"])
                    else:
                        # Yönetici özeti
                        if llm_result.get("executive_summary"):
                            st.subheader("Yönetici Özeti")
                            st.info(llm_result["executive_summary"])

                        # Segmentler arası içgörüler
                        if llm_result.get("cross_segment_insights"):
                            st.subheader("Segmentler Arası İçgörüler")
                            for insight in llm_result["cross_segment_insights"]:
                                st.markdown(f"- {insight}")

                        # Her segment için detay
                        st.subheader("Segment Detayları")
                        for seg in llm_result.get("segments", []):
                            with st.expander(f"🏷️ {seg.get('name', f'Segment {seg.get('id', '?')}')}"):
                                st.markdown(f"**Profil:** {seg.get('profile', '')}")
                                st.markdown(f"**Davranış Analizi:** {seg.get('behavioral_analysis', '')}")

                                if seg.get("key_insights"):
                                    st.markdown("**İçgörüler:**")
                                    for ins in seg["key_insights"]:
                                        st.markdown(f"- {ins}")

                                if seg.get("recommended_actions"):
                                    st.markdown("**Önerilen Aksiyonlar:**")
                                    for act in seg["recommended_actions"]:
                                        st.markdown(f"- ✅ {act}")

                                if seg.get("risk_notes"):
                                    st.markdown("**Risk / Dikkat Noktaları:**")
                                    for risk in seg["risk_notes"]:
                                        st.markdown(f"- ⚠️ {risk}")

                        # =====================================================
                        # ADIM 7: Geri Bildirim
                        # =====================================================
                        st.header("6. Geri Bildirim")
                        feedback_score = st.slider(
                            "Yorumların kalitesini puanlayın:",
                            min_value=1,
                            max_value=5,
                            value=3,
                            help="1: Çok kötü, 5: Mükemmel",
                        )
                        feedback_text = st.text_area("Ek yorumunuz (isteğe bağlı):")
                        if st.button("Geri Bildirim Gönder"):
                            st.success("Geri bildiriminiz kaydedildi. Teşekkürler!")

                except Exception as e:
                    st.error(f"Yapay zekâ yorumlama sırasında hata oluştu: {e}")
                    st.info("Lütfen Streamlit Secrets'ta LLM_API_KEY, LLM_BASE_URL ve LLM_MODEL tanımlı olduğundan emin olun.")

            # =====================================================
            # ADIM 8: Dışa Aktarma
            # =====================================================
            st.header("7. Rapor Dışa Aktarma")

            col1, col2 = st.columns(2)
            with col1:
                llm_res = st.session_state.llm_result or {}
                excel_data = export_to_excel(df, labels, profiles, llm_res, selected)
                st.download_button(
                    label="📥 Excel Raporu İndir",
                    data=excel_data,
                    file_name="segment_analiz_raporu.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            with col2:
                json_data = export_to_json(profiles, llm_res)
                st.download_button(
                    label="📥 JSON Raporu İndir",
                    data=json_data,
                    file_name="segment_analiz_raporu.json",
                    mime="application/json",
                    use_container_width=True,
                )
