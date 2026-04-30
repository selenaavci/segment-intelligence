"""
Segment Intelligence - Örnek Veri Seti Üretici

4 belirgin müşteri segmenti içeren gerçekçi banka müşteri verisi üretir.
Segmentler:
  1. Genç Dijital (18-30): Düşük gelir, yüksek mobil kullanım, düşük bakiye
  2. Orta Yaş Dengeli (30-50): Orta gelir, karma kanal, orta harcama
  3. Varlıklı Geleneksel (45-70): Yüksek gelir, şube ağırlıklı, yüksek bakiye
  4. Emekli Tutucu (60-80): Düşük-orta gelir, çok düşük dijital, yüksek tasarruf oranı
"""

import pandas as pd
import numpy as np

np.random.seed(42)

N = 300  # toplam müşteri sayısı
segment_sizes = [90, 80, 70, 60]

records = []

# --- Segment 1: Genç Dijital (90 kişi) ---
for i in range(segment_sizes[0]):
    records.append({
        "musteri_id": f"M{1001 + i}",
        "yas": np.random.randint(18, 31),
        "cinsiyet": np.random.choice(["Erkek", "Kadın"], p=[0.55, 0.45]),
        "aylik_gelir": int(np.random.normal(8000, 2000)),
        "aylik_harcama": int(np.random.normal(6500, 1500)),
        "hesap_bakiye": int(np.random.normal(5000, 3000)),
        "kredi_skoru": int(np.random.normal(580, 60)),
        "urun_sayisi": np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15]),
        "kredi_karti_var": np.random.choice(["Evet", "Hayır"], p=[0.85, 0.15]),
        "bireysel_kredi_var": np.random.choice(["Evet", "Hayır"], p=[0.3, 0.7]),
        "tercih_edilen_kanal": np.random.choice(
            ["Mobil", "Internet", "Şube", "ATM"], p=[0.55, 0.30, 0.05, 0.10]
        ),
        "aylik_islem_sayisi": int(np.random.normal(45, 12)),
        "sehir": np.random.choice(
            ["İstanbul", "Ankara", "İzmir", "Antalya", "Bursa"],
            p=[0.40, 0.20, 0.20, 0.10, 0.10],
        ),
        "musteri_olma_tarihi": pd.Timestamp("2022-01-01")
        + pd.Timedelta(days=int(np.random.uniform(0, 730))),
        "memnuniyet_puani": round(np.random.normal(3.8, 0.6), 1),
    })

# --- Segment 2: Orta Yaş Dengeli (80 kişi) ---
for i in range(segment_sizes[1]):
    records.append({
        "musteri_id": f"M{2001 + i}",
        "yas": np.random.randint(30, 51),
        "cinsiyet": np.random.choice(["Erkek", "Kadın"], p=[0.50, 0.50]),
        "aylik_gelir": int(np.random.normal(18000, 4000)),
        "aylik_harcama": int(np.random.normal(12000, 3000)),
        "hesap_bakiye": int(np.random.normal(35000, 15000)),
        "kredi_skoru": int(np.random.normal(680, 50)),
        "urun_sayisi": np.random.choice([2, 3, 4, 5], p=[0.20, 0.40, 0.30, 0.10]),
        "kredi_karti_var": np.random.choice(["Evet", "Hayır"], p=[0.92, 0.08]),
        "bireysel_kredi_var": np.random.choice(["Evet", "Hayır"], p=[0.55, 0.45]),
        "tercih_edilen_kanal": np.random.choice(
            ["Mobil", "Internet", "Şube", "ATM"], p=[0.30, 0.30, 0.25, 0.15]
        ),
        "aylik_islem_sayisi": int(np.random.normal(30, 8)),
        "sehir": np.random.choice(
            ["İstanbul", "Ankara", "İzmir", "Antalya", "Bursa"],
            p=[0.35, 0.25, 0.15, 0.15, 0.10],
        ),
        "musteri_olma_tarihi": pd.Timestamp("2018-01-01")
        + pd.Timedelta(days=int(np.random.uniform(0, 1460))),
        "memnuniyet_puani": round(np.random.normal(3.5, 0.7), 1),
    })

# --- Segment 3: Varlıklı Geleneksel (70 kişi) ---
for i in range(segment_sizes[2]):
    records.append({
        "musteri_id": f"M{3001 + i}",
        "yas": np.random.randint(45, 71),
        "cinsiyet": np.random.choice(["Erkek", "Kadın"], p=[0.60, 0.40]),
        "aylik_gelir": int(np.random.normal(40000, 10000)),
        "aylik_harcama": int(np.random.normal(20000, 5000)),
        "hesap_bakiye": int(np.random.normal(150000, 50000)),
        "kredi_skoru": int(np.random.normal(780, 40)),
        "urun_sayisi": np.random.choice([3, 4, 5, 6], p=[0.15, 0.30, 0.35, 0.20]),
        "kredi_karti_var": np.random.choice(["Evet", "Hayır"], p=[0.95, 0.05]),
        "bireysel_kredi_var": np.random.choice(["Evet", "Hayır"], p=[0.40, 0.60]),
        "tercih_edilen_kanal": np.random.choice(
            ["Mobil", "Internet", "Şube", "ATM"], p=[0.10, 0.15, 0.60, 0.15]
        ),
        "aylik_islem_sayisi": int(np.random.normal(20, 6)),
        "sehir": np.random.choice(
            ["İstanbul", "Ankara", "İzmir", "Antalya", "Bursa"],
            p=[0.45, 0.25, 0.15, 0.10, 0.05],
        ),
        "musteri_olma_tarihi": pd.Timestamp("2010-01-01")
        + pd.Timedelta(days=int(np.random.uniform(0, 2920))),
        "memnuniyet_puani": round(np.random.normal(4.2, 0.5), 1),
    })

# --- Segment 4: Emekli Tutucu (60 kişi) ---
for i in range(segment_sizes[3]):
    records.append({
        "musteri_id": f"M{4001 + i}",
        "yas": np.random.randint(60, 81),
        "cinsiyet": np.random.choice(["Erkek", "Kadın"], p=[0.45, 0.55]),
        "aylik_gelir": int(np.random.normal(12000, 3000)),
        "aylik_harcama": int(np.random.normal(5000, 1500)),
        "hesap_bakiye": int(np.random.normal(80000, 30000)),
        "kredi_skoru": int(np.random.normal(720, 45)),
        "urun_sayisi": np.random.choice([1, 2, 3], p=[0.40, 0.45, 0.15]),
        "kredi_karti_var": np.random.choice(["Evet", "Hayır"], p=[0.60, 0.40]),
        "bireysel_kredi_var": np.random.choice(["Evet", "Hayır"], p=[0.10, 0.90]),
        "tercih_edilen_kanal": np.random.choice(
            ["Mobil", "Internet", "Şube", "ATM"], p=[0.05, 0.05, 0.70, 0.20]
        ),
        "aylik_islem_sayisi": int(np.random.normal(10, 4)),
        "sehir": np.random.choice(
            ["İstanbul", "Ankara", "İzmir", "Antalya", "Bursa"],
            p=[0.30, 0.30, 0.15, 0.15, 0.10],
        ),
        "musteri_olma_tarihi": pd.Timestamp("2005-01-01")
        + pd.Timedelta(days=int(np.random.uniform(0, 3650))),
        "memnuniyet_puani": round(np.random.normal(3.2, 0.8), 1),
    })

df = pd.DataFrame(records)

# Shuffle so segments aren't in order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clamp values to realistic ranges
df["aylik_gelir"] = df["aylik_gelir"].clip(lower=4000)
df["aylik_harcama"] = df["aylik_harcama"].clip(lower=1000)
df["hesap_bakiye"] = df["hesap_bakiye"].clip(lower=0)
df["kredi_skoru"] = df["kredi_skoru"].clip(lower=300, upper=900)
df["aylik_islem_sayisi"] = df["aylik_islem_sayisi"].clip(lower=1)
df["memnuniyet_puani"] = df["memnuniyet_puani"].clip(lower=1.0, upper=5.0)

# Add some missing values (~3%) to test imputation
rng = np.random.default_rng(42)
for col in ["aylik_gelir", "hesap_bakiye", "kredi_skoru", "memnuniyet_puani"]:
    mask = rng.random(len(df)) < 0.03
    df.loc[mask, col] = np.nan

for col in ["tercih_edilen_kanal", "sehir"]:
    mask = rng.random(len(df)) < 0.02
    df.loc[mask, col] = np.nan

# Save
df.to_csv(
    "/Users/selenaavci/Desktop/segment-intelligence/sample_data/banka_musteri_ornegi.csv",
    index=False,
)
print(f"✓ {len(df)} satırlık örnek veri oluşturuldu.")
print(f"\nKolon tipleri:")
print(f"  Numerik:     yas, aylik_gelir, aylik_harcama, hesap_bakiye, kredi_skoru, urun_sayisi, aylik_islem_sayisi, memnuniyet_puani")
print(f"  Kategorik:   cinsiyet, kredi_karti_var, bireysel_kredi_var, tercih_edilen_kanal, sehir")
print(f"  Tarih:       musteri_olma_tarihi")
print(f"  ID:          musteri_id")
print(f"\nEksik değer sayıları:")
print(df.isnull().sum()[df.isnull().sum() > 0])
