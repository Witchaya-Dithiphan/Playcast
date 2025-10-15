# ======================================
# owners_gui_compact.py
# Compact GUI with Categories in Predict & Similarity
# ======================================
import sys
sys.stdout.reconfigure(encoding='utf-8')

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
from pathlib import Path
import webbrowser
from playcast import predict_new_game, find_similar_games

# --------------------------------------
# Files
# --------------------------------------
CSV_PATH = Path("games_clean.csv")
MODEL_PATH = Path("owners_model_mlr.joblib")

try:
    df = pd.read_csv(CSV_PATH)
    bundle = joblib.load(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Error", f"Cannot load model or CSV:\n{e}")
    raise SystemExit

# --------------------------------------
# Helper: extract unique lists
# --------------------------------------
def extract_unique_list(col):
    if col not in df.columns:
        return []
    vals = []
    for v in df[col].dropna().astype(str):
        for sep in [",", ";", "|", "/"]:
            if sep in v:
                vals += [x.strip() for x in v.split(sep)]
                break
        else:
            vals.append(v.strip())
    return sorted(set(v for v in vals if v and v.lower() not in {"none", "nan"}))

publishers_all = extract_unique_list("Publishers")
languages_all = extract_unique_list("Supported languages")
tags_all = extract_unique_list("Tags")
categories_all = extract_unique_list("Categories")

# --------------------------------------
# GUI setup
# --------------------------------------
root = tk.Tk()
root.title("🎮 Game Owners Predictor (with Categories)")
root.geometry("840x740")

main = ttk.Frame(root, padding=10)
main.pack(fill="both", expand=True)

# Scrollable frame
canvas = tk.Canvas(main)
scrollbar = ttk.Scrollbar(main, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)
scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

row = 0
def add_input(label, var, default="", width=20):
    global row
    ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
    var.set(default)
    ttk.Entry(scrollable_frame, textvariable=var, width=width).grid(row=row, column=1, sticky="w", pady=2)
    row += 1

# --------------------------------------
# Game Search Box
# --------------------------------------
def make_game_search_box():
    global row
    ttk.Label(scrollable_frame, text="🎮 Game:").grid(row=row, column=0, sticky="w", pady=2)
    frame = ttk.Frame(scrollable_frame)
    frame.grid(row=row, column=1, sticky="w", pady=2)
    row += 1

    var = tk.StringVar()
    entry = ttk.Entry(frame, textvariable=var, width=40)
    entry.pack(side="top", fill="x")
    listbox = tk.Listbox(frame, height=4, width=40)
    listbox.pack_forget()

    def update_list(*_):
        search = var.get().lower().strip()
        listbox.delete(0, tk.END)
        if search:
            matches = [n for n in df["Name"].dropna().astype(str) if search in n.lower()]
            if matches:
                listbox.pack(side="bottom", fill="x")
                for name in matches[:15]:
                    listbox.insert(tk.END, name)
            else:
                listbox.pack_forget()
        else:
            listbox.pack_forget()

    def on_select(event):
        sel = listbox.curselection()
        if sel:
            name = listbox.get(sel[0])
            var.set(name)
            listbox.pack_forget()
            fill_game_data(name)

    var.trace_add("write", update_list)
    listbox.bind("<Double-1>", on_select)
    return var

def fill_game_data(name):
    """Auto-fill fields when a game is selected."""
    try:
        row_data = df[df["Name"].str.lower() == name.lower()].iloc[0]
        price_var.set(row_data.get("Price", ""))
        year_var.set(str(row_data.get("Release date", ""))[-4:])
        playtime_var.set(row_data.get("Average playtime forever", ""))
        pos_var.set(row_data.get("Positive", ""))
        neg_var.set(row_data.get("Negative", ""))
        pub_var.set(str(row_data.get("Publishers", "")).split(",")[0].strip())
        tag_var.set(row_data.get("Tags", ""))
        lang_var.set(row_data.get("Supported languages", ""))
        cat_var.set(row_data.get("Categories", ""))  # ✅ NEW
    except Exception as e:
        messagebox.showerror("Error", f"❌ Could not load game data:\n{e}")

game_var = make_game_search_box()

# --------------------------------------
# Searchable Publisher (single)
# --------------------------------------
def make_searchable_publisher(label, options):
    global row
    ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
    frame = ttk.Frame(scrollable_frame)
    frame.grid(row=row, column=1, sticky="w", pady=2)
    row += 1
    var = tk.StringVar()
    entry = ttk.Entry(frame, textvariable=var, width=40)
    entry.pack(side="top", fill="x")
    listbox = tk.Listbox(frame, height=4, width=40)
    listbox.pack_forget()

    def update_list(*_):
        s = var.get().lower()
        listbox.delete(0, tk.END)
        if s:
            for o in options:
                if s in o.lower():
                    listbox.insert(tk.END, o)
            listbox.pack(side="bottom", fill="x")
        else:
            listbox.pack_forget()

    def on_select(event):
        sel = listbox.curselection()
        if sel:
            chosen = listbox.get(sel[0])
            var.set(chosen)
            listbox.pack_forget()

    var.trace_add("write", update_list)
    listbox.bind("<Double-1>", on_select)
    return var

# --------------------------------------
# Multi-Select Searchable Dropdown
# --------------------------------------
def make_multi_select_searchable(label, options):
    global row
    ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
    frame = ttk.Frame(scrollable_frame)
    frame.grid(row=row, column=1, sticky="w", pady=2)
    row += 1
    
    var = tk.StringVar()
    entry = ttk.Entry(frame, textvariable=var, width=40)
    entry.pack(side="top", fill="x")
    listbox = tk.Listbox(frame, height=6, width=40)
    listbox.pack_forget()
    
    def update_list(*_):
        current_text = var.get()
        # Get the search term (text after last comma)
        if "," in current_text:
            search_term = current_text.split(",")[-1].strip().lower()
        else:
            search_term = current_text.strip().lower()
        
        listbox.delete(0, tk.END)
        if search_term:
            matches = [o for o in options if search_term in o.lower()]
            if matches:
                listbox.pack(side="bottom", fill="x")
                for match in matches[:20]:
                    listbox.insert(tk.END, match)
            else:
                listbox.pack_forget()
        else:
            listbox.pack_forget()
    
    def on_select(event):
        sel = listbox.curselection()
        if sel:
            chosen = listbox.get(sel[0])
            current_text = var.get()
            
            # Split by comma and remove the last incomplete part
            if "," in current_text:
                parts = [p.strip() for p in current_text.split(",")[:-1]]
                # Add the new selection if not already present
                if chosen not in parts:
                    parts.append(chosen)
                var.set(", ".join(parts) + ", ")
            else:
                var.set(chosen + ", ")
            
            listbox.pack_forget()
            entry.focus_set()
            entry.icursor(tk.END)
    
    var.trace_add("write", update_list)
    listbox.bind("<Double-1>", on_select)
    return var

pub_var = make_searchable_publisher("🏢 Publisher:", publishers_all)
lang_var = make_multi_select_searchable("🌐 Languages (,):", languages_all)
tag_var = make_multi_select_searchable("🏷️ Tags (,):", tags_all)
cat_var = make_multi_select_searchable("🗂 Categories (,):", categories_all)

price_var = tk.StringVar()
add_input("💰 Price ($):", price_var, "9.99")
year_var = tk.StringVar()
add_input("📅 Year:", year_var, "2015")
playtime_var = tk.StringVar()
add_input("⏱ Playtime:", playtime_var, "100")
pos_var = tk.StringVar()
add_input("👍 Positive:", pos_var, "1000")
neg_var = tk.StringVar()
add_input("👎 Negative:", neg_var, "100")

# --------------------------------------
# Validation
# --------------------------------------
def parse_commas(s):
    return [x.strip() for x in s.split(",") if x.strip()]

def validate_inputs():
    try:
        year = int(year_var.get())
        if year < 1920:
            messagebox.showerror("Error", "📅 Year must be ≥1920")
            return False
    except:
        messagebox.showerror("Error", "Invalid year format")
        return False
    return True

# --------------------------------------
# Predict / Similar Buttons
# --------------------------------------
def do_predict():
    if not validate_inputs():
        return
    try:
        result = predict_new_game(
            model_bundle_path=MODEL_PATH,
            price_num=float(price_var.get()),
            release_year=int(year_var.get()),
            languages=parse_commas(lang_var.get()),
            tags=parse_commas(tag_var.get()),
            publishers=[pub_var.get().strip()],
            categories=parse_commas(cat_var.get()),  # ✅ NEW
            avg_playtime=float(playtime_var.get()),
            positive=int(pos_var.get()),
            negative=int(neg_var.get())
        )
        messagebox.showinfo("Result", f"🎯 Estimated Owners ≈ {result:,.0f}")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{e}")

def do_find_similar():
    if not validate_inputs():
        return
    try:
        df_sim = find_similar_games(
            model_bundle_path=MODEL_PATH,
            dataset_csv_path=CSV_PATH,
            tags=parse_commas(tag_var.get()),
            languages=parse_commas(lang_var.get()),
            categories=parse_commas(cat_var.get()),  # ✅ NEW
            top_k=10,
        )

        popup = tk.Toplevel(root)
        popup.title("🧩 Similar Games")
        popup.geometry("640x300")
        cols = list(df_sim.columns)
        tree = ttk.Treeview(popup, columns=cols, show="headings", height=10)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=180)
        for _, r in df_sim.iterrows():
            tree.insert("", "end", values=list(r))
        tree.pack(fill="both", expand=True)

        def on_click(e):
            item = tree.focus()
            if item:
                vals = tree.item(item, "values")
                if len(vals) > 1:
                    webbrowser.open_new_tab(f"https://www.google.com/search?q={vals[1]}")
        tree.bind("<Double-1>", on_click)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to find similar games:\n{e}")

# --------------------------------------
# Buttons
# --------------------------------------
ttk.Button(scrollable_frame, text="🔮 Predict", command=do_predict).grid(row=row, column=0, pady=8)
ttk.Button(scrollable_frame, text="🧩 Similar Games", command=do_find_similar).grid(row=row, column=1, pady=8)
row += 1

root.mainloop()