import re
import sys
import pandas as pd
from pathlib import Path

# คอนโซล Windows (cp1252) เข้ารหัส emoji ไม่ได้ → บังคับเป็น utf-8 กัน crash
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]   # โฟลเดอร์ราก playcast/
DATA_DIR = BASE_DIR / "data"
input_csv = str(DATA_DIR / "converted.csv")
output_csv = str(DATA_DIR / "games_clean.csv")

non_game_keywords = {
    "categories": ["SteamVR Tool", "Application", "Utilities", "Design & Illustration",
                   "Education", "Software Training", "Audio Production", "Video Production"],
    "genres": ["Utilities", "Design & Illustration", "Education",
               "Software Training", "Audio Production", "Video Production",
               "Animation & Modeling", "Free to Play", "Early Access", "Free To Play"]
}

df = pd.read_csv(input_csv)
print(f"📊 ขนาดข้อมูลดิบทั้งหมด: {len(df):,} แถว\n")

cat_pat = "|".join(map(re.escape, non_game_keywords["categories"]))
gen_pat = "|".join(map(re.escape, non_game_keywords["genres"]))

df["Price_num"] = pd.to_numeric(df["Price"].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")
df["Avg_play"] = pd.to_numeric(df.get("Average playtime forever", 0), errors="coerce")
df["Med_play"] = pd.to_numeric(df.get("Median playtime forever", 0), errors="coerce")

def report(condition, description):
    removed = len(df) - condition.sum()
    print(f"❌ {description:<55} {removed:>6,} แถวถูกตัดออก")
    return condition

mask = pd.Series(True, index=df.index)

# ---------- การตัดข้อมูลที่ไม่ใช่เกม ----------
mask &= report(~df["Categories"].fillna("").str.contains(cat_pat, regex=True, na=False),
               "ไม่อยู่ใน Categories ที่เป็นโปรแกรม/เครื่องมือ")

mask &= report(~df["Genres"].fillna("").str.contains(gen_pat, regex=True, na=False),
               "ไม่อยู่ใน Genres ที่เป็นโปรแกรม/เครื่องมือ")

# ---------- การกรองราคา ----------
mask &= report(df["Price_num"].ge(1.0),
               "ราคา >= 1.0 USD")

# ---------- ตรวจ Windows ----------
if df["Windows"].dtype == bool:
    mask &= report(df["Windows"] == True, "รองรับ Windows")
else:
    mask &= report(df["Windows"].fillna("").astype(str).str.contains("Windows", case=False, na=False),
                   "รองรับ Windows (string)")

# ---------- ตรวจเวลาเล่น ----------
mask &= report(df["Avg_play"].gt(0), "Average playtime > 0")
mask &= report(df["Med_play"].gt(0), "Median playtime > 0")

# ---------- ตรวจข้อมูลสำคัญ ----------
mask &= report(df["Name"].notna(), "มีชื่อเกม (Name)")
mask &= report(df["Developers"].notna(), "มี Developers")
mask &= report(df["Publishers"].notna(), "มี Publishers")
mask &= report(df["Categories"].notna(), "มี Categories")
mask &= report(df["Genres"].notna(), "มี Genres")

# ---------- ตรวจภาษา ----------
mask &= report(df["Supported languages"].fillna("").str.contains("English", case=False, na=False),
               "รองรับภาษาอังกฤษ")

# ---------- ตรวจ Tags ----------
mask &= report(df["Tags"].fillna("").astype(str).str.strip().ne(""), "มี Tags ไม่ว่าง")
mask &= report(~df["Tags"].astype(str).isin(["[]", "nan", "None", "none", "NaN"]),
               "Tags ไม่ใช่ค่าหลอก (nan/None/[])")

# ---------- สร้าง dataset ที่คลีนแล้ว ----------
df_clean = df[mask].copy()
print("\n✅ สรุปผลหลังกรอง:")
print(f"เหลือข้อมูลทั้งหมด {len(df_clean):,} จาก {len(df):,} แถว")

df_clean.to_csv(output_csv, index=False)
print(f"💾 บันทึกข้อมูลสะอาดแล้วที่ '{output_csv}'")
