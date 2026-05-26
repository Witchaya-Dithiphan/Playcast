# 🎮 PlayCast — ทำนายยอดผู้เป็นเจ้าของเกมบน Steam

> **โปรเจกต์ปี 2 วิชา Linear Algebra**
> นำทฤษฎี **เวกเตอร์ (Vector)** และ **เมทริกซ์ (Matrix)** มาประยุกต์กับ Machine Learning
> เพื่อทำนายจำนวนผู้เป็นเจ้าของ (estimated owners) ของเกมใหม่ที่จะวางขายบน Steam

---

## 1. โปรเจกต์นี้ทำอะไร

ผู้พัฒนาเกมอยากรู้ว่า "เกมที่กำลังจะปล่อย" น่าจะมีคนซื้อ/เป็นเจ้าของประมาณเท่าไหร่
PlayCast รับข้อมูลคุณสมบัติของเกม (ราคา, ปีที่วางขาย, แท็ก, ภาษา, ผู้จัดจำหน่าย, หมวดหมู่ ฯลฯ)
แล้วทำนายจำนวนผู้เป็นเจ้าของโดยประมาณ พร้อมแนะนำ "เกมที่คล้ายกัน" ในตลาด

มีหน้าจอ GUI ให้กรอกข้อมูลและกดทำนายได้ทันที

---

## 2. หลักการทำงาน (เชื่อมกับ Linear Algebra)

หัวใจของโปรเจกต์คือการแปลงข้อมูลเกมให้อยู่ในรูป **เวกเตอร์** แล้วประมวลผลด้วย **เมทริกซ์**

| แนวคิด Linear Algebra | ใช้ตรงไหนในโปรเจกต์ |
|---|---|
| **เวกเตอร์ One-hot** | แปลงข้อมูลข้อความ (Tags / Languages / Publishers / Categories) เป็นเวกเตอร์ฐานสอง ด้วย `MultiLabelBinarizer` — เกมแต่ละเกมกลายเป็นเวกเตอร์ในปริภูมิ 4,963 มิติ |
| **เมทริกซ์ฟีเจอร์ X** | นำเวกเตอร์ของทุกเกมมาเรียงเป็นเมทริกซ์ขนาด (จำนวนเกม × 4,963) |
| **Normal Equation / Least Squares** | `LinearRegression` หาค่าสัมประสิทธิ์ **β** ที่ทำให้ Xβ ≈ y โดยแก้สมการเชิงเส้น (พื้นฐานคือ β = (XᵀX)⁻¹Xᵀy) |
| **Dot product & Cosine Similarity** | วัดความคล้ายของเกมจากมุมระหว่างเวกเตอร์ฟีเจอร์ → ใช้แนะนำเกมที่คล้ายกัน |
| **Log transform ของ target** | ทำนาย `log1p(owners)` แล้วแปลงกลับด้วย `expm1` เพื่อให้ข้อมูลที่เบ้มากกระจายเหมาะกับโมเดลเชิงเส้น |

ขั้นตอนการประมวลผลข้อมูล (data pipeline):

```
games.json ──▶ converted.csv ──▶ games_clean.csv ──▶ owners_model_mlr.joblib ──▶ GUI ทำนาย
 (Kaggle)      แปลงเป็นตาราง       กรองให้เหลือเกมจริง    โมเดลที่เทรนแล้ว         + แนะนำเกมคล้าย
              convert_json_csv.py  clean_data.py        playcast.py            gui_play_cast.py
```

---

## 3. โครงสร้างโปรเจกต์

```
playcast/
├── README.md                    เอกสารนี้
├── CLAUDE.md                    สรุปโปรเจกต์เชิงเทคนิค (สำหรับผู้รีวิว/HR)
├── .gitignore
│
├── src/                         ซอร์สโค้ดทั้งหมด
│   ├── convert_json_csv.py      1) แปลง games.json (Kaggle) → converted.csv
│   ├── clean_data.py            2) กรองข้อมูลที่ไม่ใช่เกม → games_clean.csv
│   ├── playcast.py              3) เทรนโมเดล + ฟังก์ชันทำนาย/หาเกมคล้าย
│   └── gui_play_cast.py         4) หน้าจอ tkinter สำหรับกรอกข้อมูลและทำนาย
│
├── data/                        ข้อมูล
│   ├── games_clean.csv          ชุดข้อมูลที่ใช้เทรนจริง (~10,400 เกม) — อยู่ใน git
│   ├── games.json               ต้นทางจาก Kaggle (โหลดมาวางเอง — ไม่อยู่ใน git, 707MB)
│   └── converted.csv            ไฟล์กลางที่ pipeline สร้าง (ไม่อยู่ใน git, 505MB)
│
├── models/
│   └── owners_model_mlr.joblib  โมเดลที่เทรนแล้ว (เก็บ model + encoders + feature_columns)
│
└── reference/                   รายการค่าที่พบในข้อมูล (ผล EDA) ไว้ดูเป็นตัวเลือก/อ้างอิงในรายงาน
    └── *_unique.csv / *_counts.csv   (tags, categories, genres, publishers, languages)
```

---

## 4. วิธีติดตั้งและใช้งาน

### ติดตั้งไลบรารี
```bash
pip install numpy pandas scikit-learn joblib
```

### ใช้งานทันที (มีโมเดลให้แล้ว)
ในโปรเจกต์มี `models/owners_model_mlr.joblib` และ `data/games_clean.csv` อยู่แล้ว เปิด GUI ได้เลย:
```bash
python src/gui_play_cast.py
```
กรอกราคา/ปี/แท็ก/ภาษา/หมวดหมู่ แล้วกด **🔮 Predict** เพื่อดูยอดผู้เป็นเจ้าของโดยประมาณ
หรือกด **🧩 Similar Games** เพื่อดูเกมที่คล้ายกัน

### เทรนโมเดลใหม่จากศูนย์ (ถ้าต้องการ)
```bash
# 0) โหลด games.json จาก Kaggle (Steam Games Dataset) มาวางไว้ที่ data/games.json

python src/convert_json_csv.py   # 1) games.json → data/converted.csv
python src/clean_data.py         # 2) converted.csv → data/games_clean.csv
python src/playcast.py           # 3) เทรน → models/owners_model_mlr.joblib
python src/gui_play_cast.py      # 4) เปิดหน้าจอทำนาย
```

> ทุกสคริปต์อ้างอิง path จากตำแหน่งไฟล์ตัวเอง (`__file__`) จึงรันจากโฟลเดอร์ไหนก็ได้

---

## 5. ขอบเขต (Scope)

**ทำได้:**
- รับข้อมูลดิบจาก Kaggle (JSON) → ทำความสะอาด → เทรนโมเดล Linear Regression
- ทำนายยอดผู้เป็นเจ้าของของเกมจากคุณสมบัติ (ราคา, ปี, เวลาเล่นเฉลี่ย, รีวิว, แท็ก, ภาษา, ผู้จัดจำหน่าย, หมวดหมู่)
- แนะนำเกมที่คล้ายกันด้วย cosine similarity
- หน้าจอ GUI ค้นหาเกม/เติมข้อมูลอัตโนมัติ/ทำนาย

**ข้อจำกัด:**
- เป็นโมเดลเชิงเส้น (OLS) ใช้สาธิตทฤษฎี Linear Algebra ไม่ได้มุ่งความแม่นยำระดับ production
- โมเดลใช้จำนวนรีวิว (Positive/Negative) เป็นฟีเจอร์ ซึ่งเกมใหม่จริง ๆ ยังไม่มี — ตอนทำนายต้องใส่ค่าประมาณการ
- ข้อมูลเป็น snapshot จาก Kaggle ไม่ได้อัปเดตเรียลไทม์

---

## 6. ข้อมูลต้นทาง

**Steam Games Dataset (Kaggle)** — ไฟล์ JSON ที่ key ด้วย AppID ของแต่ละเกม
หลังกรองด้วย `clean_data.py` (ตัดซอฟต์แวร์/เครื่องมือที่ไม่ใช่เกม, ราคา < 1 USD, ไม่มีเวลาเล่น ฯลฯ)
เหลือ ~10,400 เกมที่ใช้เทรน
