import datetime as dt
import math
import os
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


APP_DIR = os.path.join(os.path.expanduser("~"), ".weight_tracker")
DB_PATH = os.path.join(APP_DIR, "measurements.sqlite")
DATE_FORMAT = "%Y-%m-%d"
MAINTENANCE_BASELINE_DEFAULT = 2200.0
WAIST_TO_FAT_KG_PER_CM = 0.25


matplotlib.use("TkAgg")


def ensure_db():
    os.makedirs(APP_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS measurements (
            date TEXT PRIMARY KEY,
            weight REAL,
            waist REAL,
            calories REAL,
            calories_accuracy REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            height_cm REAL,
            age_years INTEGER,
            sex TEXT,
            activity_level TEXT,
            baseline_weight REAL,
            baseline_maintenance REAL
        )
        """
    )
    migrate_measurements(conn)
    conn.commit()
    return conn


def migrate_measurements(conn):
    columns = [row[1] for row in conn.execute("PRAGMA table_info(measurements)")]
    if "calories" not in columns:
        conn.execute("ALTER TABLE measurements ADD COLUMN calories REAL")
    if "calories_accuracy" not in columns:
        conn.execute("ALTER TABLE measurements ADD COLUMN calories_accuracy REAL")


def upsert_measurement(conn, date_str, weight, waist, calories, calories_accuracy):
    conn.execute(
        """
        INSERT INTO measurements (date, weight, waist, calories, calories_accuracy)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            weight = excluded.weight,
            waist = excluded.waist,
            calories = excluded.calories,
            calories_accuracy = excluded.calories_accuracy
        """,
        (date_str, weight, waist, calories, calories_accuracy),
    )
    conn.commit()


def delete_measurement(conn, date_str):
    conn.execute("DELETE FROM measurements WHERE date = ?", (date_str,))
    conn.commit()


def fetch_measurements(conn):
    cur = conn.execute(
        "SELECT date, weight, waist, calories, calories_accuracy FROM measurements ORDER BY date"
    )
    rows = cur.fetchall()
    return rows


def fetch_profile(conn):
    cur = conn.execute(
        """
        SELECT height_cm, age_years, sex, activity_level, baseline_weight, baseline_maintenance
        FROM profile
        WHERE id = 1
        """
    )
    row = cur.fetchone()
    if row:
        return {
            "height_cm": row[0],
            "age_years": row[1],
            "sex": row[2],
            "activity_level": row[3],
            "baseline_weight": row[4],
            "baseline_maintenance": row[5],
        }
    return {
        "height_cm": None,
        "age_years": None,
        "sex": "male",
        "activity_level": "moderate",
        "baseline_weight": None,
        "baseline_maintenance": MAINTENANCE_BASELINE_DEFAULT,
    }


def save_profile(conn, profile):
    conn.execute(
        """
        INSERT INTO profile (
            id, height_cm, age_years, sex, activity_level, baseline_weight, baseline_maintenance
        )
        VALUES (1, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            height_cm = excluded.height_cm,
            age_years = excluded.age_years,
            sex = excluded.sex,
            activity_level = excluded.activity_level,
            baseline_weight = excluded.baseline_weight,
            baseline_maintenance = excluded.baseline_maintenance
        """,
        (
            profile["height_cm"],
            profile["age_years"],
            profile["sex"],
            profile["activity_level"],
            profile["baseline_weight"],
            profile["baseline_maintenance"],
        ),
    )
    conn.commit()


def parse_date(date_str):
    return dt.datetime.strptime(date_str, DATE_FORMAT).date()


def format_date(date_obj):
    return date_obj.strftime(DATE_FORMAT)


def build_series(rows, column_index):
    if not rows:
        return [], []

    dates = [parse_date(row[0]) for row in rows]
    values = [row[column_index] for row in rows]

    min_date = min(dates)
    max_date = max(dates)
    full_dates = []
    full_values = []

    date_to_value = {d: v for d, v in zip(dates, values)}

    current = min_date
    while current <= max_date:
        full_dates.append(current)
        full_values.append(date_to_value.get(current))
        current += dt.timedelta(days=1)

    return full_dates, full_values


def interpolate_series(values):
    if not values:
        return []

    indexed = [
        (i, v)
        for i, v in enumerate(values)
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not indexed:
        return [np.nan for _ in values]

    x_known = np.array([i for i, _ in indexed], dtype=float)
    y_known = np.array([v for _, v in indexed], dtype=float)
    x_all = np.arange(len(values), dtype=float)

    if len(x_known) == 1:
        return [y_known[0] for _ in values]

    interpolated = np.interp(x_all, x_known, y_known)
    return interpolated.tolist()


def moving_average(values, window=7):
    if len(values) < window:
        return [np.nan for _ in values]

    kernel = np.ones(window) / window
    conv = np.convolve(values, kernel, mode="valid")
    prefix = [np.nan] * (window - 1)
    return prefix + conv.tolist()


def activity_multiplier(level):
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9,
    }
    return multipliers.get(level, 1.55)


def calculate_baseline_maintenance(profile, weight_kg):
    height_cm = profile.get("height_cm")
    age_years = profile.get("age_years")
    sex = profile.get("sex", "male")
    if height_cm and age_years and weight_kg:
        if sex == "female":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5
        return bmr * activity_multiplier(profile.get("activity_level", "moderate"))
    baseline = profile.get("baseline_maintenance")
    if baseline:
        return baseline
    return MAINTENANCE_BASELINE_DEFAULT


def calculate_maintenance_range(
    dates,
    calories,
    calories_accuracy,
    weight_values,
    waist_values,
    profile,
):
    if not dates:
        return [], [], []

    weight_interp = interpolate_series(weight_values)
    waist_interp = interpolate_series(waist_values)
    weight_smoothed = moving_average(weight_interp)
    waist_smoothed = moving_average(waist_interp)
    calories_interp = interpolate_series(calories)
    accuracy_interp = interpolate_series(calories_accuracy)

    maintenance_low = []
    maintenance_high = []
    maintenance_mid = []

    total_days = len(dates)
    tracked_days = len([v for v in calories if v is not None])
    coverage = tracked_days / total_days if total_days else 0

    baseline_weight = profile.get("baseline_weight")
    latest_weight = next((v for v in reversed(weight_interp) if not math.isnan(v)), None)
    weight_for_baseline = latest_weight if latest_weight else baseline_weight
    baseline = calculate_baseline_maintenance(profile, weight_for_baseline)

    correction = 0.0
    learning_rate = 0.2

    for index in range(total_days):
        if index == 0:
            maintenance_low.append(baseline)
            maintenance_high.append(baseline)
            maintenance_mid.append(baseline)
            continue

        weight_today = weight_smoothed[index]
        weight_yesterday = weight_smoothed[index - 1]
        waist_today = waist_smoothed[index]
        waist_yesterday = waist_smoothed[index - 1]

        if math.isnan(weight_today) or math.isnan(weight_yesterday):
            weight_change = 0
        else:
            weight_change = weight_today - weight_yesterday
        if math.isnan(waist_today) or math.isnan(waist_yesterday):
            waist_change = 0
        else:
            waist_change = waist_today - waist_yesterday
        fat_change = (weight_change + waist_change * WAIST_TO_FAT_KG_PER_CM) / 2
        deficit = -fat_change * 7700

        intake = calories_interp[index]
        accuracy = accuracy_interp[index]
        if math.isnan(intake) or math.isnan(accuracy):
            estimate_low = baseline + correction
            estimate_high = baseline + correction
            estimate_mid = baseline + correction
        else:
            intake_low = intake - accuracy
            intake_high = intake + accuracy
            estimate_low = intake_low + deficit + correction
            estimate_high = intake_high + deficit + correction
            estimate_mid = (estimate_low + estimate_high) / 2

            predicted_change = (intake - estimate_mid) / 7700
            error_kg = weight_change - predicted_change
            correction += error_kg * 7700 * learning_rate

        blended_low = baseline * (1 - coverage) + estimate_low * coverage
        blended_high = baseline * (1 - coverage) + estimate_high * coverage
        blended_mid = baseline * (1 - coverage) + estimate_mid * coverage

        maintenance_low.append(blended_low)
        maintenance_high.append(blended_high)
        maintenance_mid.append(blended_mid)

    return maintenance_low, maintenance_high, maintenance_mid


class WeightTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Weight Tracker")
        self.geometry("1100x700")
        self.minsize(1000, 650)

        self.conn = ensure_db()
        self.profile = fetch_profile(self.conn)

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TButton", padding=6)
        self.style.configure("Treeview", rowheight=24)
        self.style.configure("Heading", font=("Helvetica", 11, "bold"))

        self.create_widgets()
        self.refresh_table()
        self.refresh_charts()

    def create_widgets(self):
        root_frame = ttk.Frame(self, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(root_frame)
        top_frame.pack(fill=tk.X, pady=(0, 12))

        form_frame = ttk.LabelFrame(top_frame, text="Daily entry", padding=12)
        form_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12))

        table_frame = ttk.LabelFrame(top_frame, text="History", padding=12)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(form_frame, text="Date (YYYY-MM-DD)").grid(row=0, column=0, sticky=tk.W)
        self.date_var = tk.StringVar(value=format_date(dt.date.today()))
        self.date_entry = ttk.Entry(form_frame, textvariable=self.date_var, width=16)
        self.date_entry.grid(row=1, column=0, sticky=tk.W, pady=(0, 8))

        ttk.Label(form_frame, text="Weight (kg)").grid(row=0, column=1, sticky=tk.W)
        self.weight_var = tk.StringVar()
        self.weight_entry = ttk.Entry(form_frame, textvariable=self.weight_var, width=12)
        self.weight_entry.grid(row=1, column=1, sticky=tk.W, padx=(12, 0), pady=(0, 8))

        ttk.Label(form_frame, text="Waist (cm)").grid(row=0, column=2, sticky=tk.W)
        self.waist_var = tk.StringVar()
        self.waist_entry = ttk.Entry(form_frame, textvariable=self.waist_var, width=12)
        self.waist_entry.grid(row=1, column=2, sticky=tk.W, padx=(12, 0), pady=(0, 8))

        ttk.Label(form_frame, text="Calories").grid(row=0, column=3, sticky=tk.W)
        self.calories_var = tk.StringVar()
        self.calories_entry = ttk.Entry(form_frame, textvariable=self.calories_var, width=12)
        self.calories_entry.grid(row=1, column=3, sticky=tk.W, padx=(12, 0), pady=(0, 8))

        ttk.Label(form_frame, text="± Accuracy").grid(row=0, column=4, sticky=tk.W)
        self.calories_accuracy_var = tk.StringVar()
        self.calories_accuracy_entry = ttk.Entry(
            form_frame, textvariable=self.calories_accuracy_var, width=10
        )
        self.calories_accuracy_entry.grid(
            row=1, column=4, sticky=tk.W, padx=(12, 0), pady=(0, 8)
        )

        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=2, column=0, columnspan=5, sticky=tk.W, pady=(8, 0))

        save_button = ttk.Button(button_frame, text="Save entry", command=self.save_entry)
        save_button.pack(side=tk.LEFT)

        clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_form)
        clear_button.pack(side=tk.LEFT, padx=(8, 0))

        today_button = ttk.Button(button_frame, text="Today", command=self.set_today)
        today_button.pack(side=tk.LEFT, padx=(8, 0))

        delete_button = ttk.Button(button_frame, text="Delete day", command=self.delete_entry)
        delete_button.pack(side=tk.LEFT, padx=(8, 0))

        note = ttk.Label(
            form_frame,
            text=(
                "Body fat index is calculated as waist / weight. "
                "Maintenance calories use weight + waist changes."
            ),
            foreground="#555555",
        )
        note.grid(row=3, column=0, columnspan=5, sticky=tk.W, pady=(10, 0))

        profile_frame = ttk.LabelFrame(root_frame, text="Baseline profile", padding=12)
        profile_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(profile_frame, text="Height (cm)").grid(row=0, column=0, sticky=tk.W)
        self.height_var = tk.StringVar(value=self.profile.get("height_cm") or "")
        ttk.Entry(profile_frame, textvariable=self.height_var, width=10).grid(
            row=1, column=0, sticky=tk.W, pady=(0, 8)
        )

        ttk.Label(profile_frame, text="Age").grid(row=0, column=1, sticky=tk.W)
        self.age_var = tk.StringVar(value=self.profile.get("age_years") or "")
        ttk.Entry(profile_frame, textvariable=self.age_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=(12, 0), pady=(0, 8)
        )

        ttk.Label(profile_frame, text="Sex").grid(row=0, column=2, sticky=tk.W)
        self.sex_var = tk.StringVar(value=self.profile.get("sex") or "male")
        sex_combo = ttk.Combobox(
            profile_frame, textvariable=self.sex_var, values=["male", "female"], width=10
        )
        sex_combo.grid(row=1, column=2, sticky=tk.W, padx=(12, 0), pady=(0, 8))
        sex_combo.state(["readonly"])

        ttk.Label(profile_frame, text="Activity").grid(row=0, column=3, sticky=tk.W)
        self.activity_var = tk.StringVar(value=self.profile.get("activity_level") or "moderate")
        activity_combo = ttk.Combobox(
            profile_frame,
            textvariable=self.activity_var,
            values=["sedentary", "light", "moderate", "active", "very active"],
            width=12,
        )
        activity_combo.grid(row=1, column=3, sticky=tk.W, padx=(12, 0), pady=(0, 8))
        activity_combo.state(["readonly"])

        ttk.Label(profile_frame, text="Baseline weight (kg)").grid(
            row=0, column=4, sticky=tk.W
        )
        self.baseline_weight_var = tk.StringVar(value=self.profile.get("baseline_weight") or "")
        ttk.Entry(profile_frame, textvariable=self.baseline_weight_var, width=12).grid(
            row=1, column=4, sticky=tk.W, padx=(12, 0), pady=(0, 8)
        )

        ttk.Label(profile_frame, text="Baseline maintenance").grid(
            row=0, column=5, sticky=tk.W
        )
        self.baseline_maintenance_var = tk.StringVar(
            value=self.profile.get("baseline_maintenance") or MAINTENANCE_BASELINE_DEFAULT
        )
        ttk.Entry(profile_frame, textvariable=self.baseline_maintenance_var, width=12).grid(
            row=1, column=5, sticky=tk.W, padx=(12, 0), pady=(0, 8)
        )

        save_profile_button = ttk.Button(
            profile_frame, text="Save profile", command=self.save_profile
        )
        save_profile_button.grid(row=1, column=6, sticky=tk.W, padx=(12, 0), pady=(0, 8))

        self.tree = ttk.Treeview(
            table_frame,
            columns=("date", "weight", "waist", "calories", "accuracy"),
            show="headings",
            height=8,
        )
        self.tree.heading("date", text="Date")
        self.tree.heading("weight", text="Weight (kg)")
        self.tree.heading("waist", text="Waist (cm)")
        self.tree.heading("calories", text="Calories")
        self.tree.heading("accuracy", text="± Accuracy")
        self.tree.column("date", width=110)
        self.tree.column("weight", width=110)
        self.tree.column("waist", width=110)
        self.tree.column("calories", width=110)
        self.tree.column("accuracy", width=100)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        notebook = ttk.Notebook(root_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.weight_tab = ttk.Frame(notebook)
        self.waist_tab = ttk.Frame(notebook)
        self.bodyfat_tab = ttk.Frame(notebook)
        self.calories_tab = ttk.Frame(notebook)
        self.maintenance_tab = ttk.Frame(notebook)

        notebook.add(self.weight_tab, text="Weight")
        notebook.add(self.waist_tab, text="Waist")
        notebook.add(self.bodyfat_tab, text="Body fat index")
        notebook.add(self.calories_tab, text="Calories")
        notebook.add(self.maintenance_tab, text="Maintenance range")

        self.weight_fig, self.weight_ax = self.create_chart(self.weight_tab)
        self.waist_fig, self.waist_ax = self.create_chart(self.waist_tab)
        self.bodyfat_fig, self.bodyfat_ax = self.create_chart(self.bodyfat_tab)
        self.calories_fig, self.calories_ax = self.create_chart(self.calories_tab)
        self.maintenance_fig, self.maintenance_ax = self.create_chart(self.maintenance_tab)

    def create_chart(self, parent):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.grid(True, linestyle="--", alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        fig.canvas = canvas
        return fig, ax

    def refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        rows = fetch_measurements(self.conn)
        for date_str, weight, waist, calories, accuracy in rows:
            weight_text = f"{weight:.1f}" if weight is not None else ""
            waist_text = f"{waist:.1f}" if waist is not None else ""
            calories_text = f"{calories:.0f}" if calories is not None else ""
            accuracy_text = f"{accuracy:.0f}" if accuracy is not None else ""
            self.tree.insert(
                "",
                tk.END,
                values=(date_str, weight_text, waist_text, calories_text, accuracy_text),
            )

    def refresh_charts(self):
        rows = fetch_measurements(self.conn)
        weight_dates, weight_values = build_series(rows, 1)
        waist_dates, waist_values = build_series(rows, 2)
        calories_dates, calories_values = build_series(rows, 3)
        _, accuracy_values = build_series(rows, 4)

        self.plot_metric(
            self.weight_ax,
            self.weight_fig,
            weight_dates,
            weight_values,
            "Weight (kg)",
            color="#3b82f6",
        )
        self.plot_metric(
            self.waist_ax,
            self.waist_fig,
            waist_dates,
            waist_values,
            "Waist (cm)",
            color="#10b981",
        )

        body_dates = weight_dates if len(weight_dates) >= len(waist_dates) else waist_dates
        if body_dates:
            weight_interp = interpolate_series(weight_values)
            waist_interp = interpolate_series(waist_values)
            bodyfat = []
            for weight, waist in zip(weight_interp, waist_interp):
                if weight is None or waist is None:
                    bodyfat.append(np.nan)
                elif isinstance(weight, float) and math.isnan(weight):
                    bodyfat.append(np.nan)
                elif isinstance(waist, float) and math.isnan(waist):
                    bodyfat.append(np.nan)
                else:
                    bodyfat.append(waist / weight)
        else:
            bodyfat = []

        self.plot_metric(
            self.bodyfat_ax,
            self.bodyfat_fig,
            body_dates,
            bodyfat,
            "Body fat index (waist / weight)",
            color="#f97316",
        )

        self.plot_metric(
            self.calories_ax,
            self.calories_fig,
            calories_dates,
            calories_values,
            "Calories",
            color="#9333ea",
        )

        maintenance_dates = weight_dates if weight_dates else calories_dates
        low, high, mid = calculate_maintenance_range(
            maintenance_dates,
            calories_values,
            accuracy_values,
            weight_values,
            waist_values,
            self.profile,
        )
        self.plot_maintenance_range(maintenance_dates, low, high, mid)

    def plot_metric(self, ax, fig, dates, values, label, color):
        ax.clear()
        ax.grid(True, linestyle="--", alpha=0.3)
        if not dates:
            ax.set_title("No data yet")
            fig.canvas.draw()
            return

        interpolated = interpolate_series(values)
        average = moving_average(interpolated)

        x_vals = [dt.datetime.combine(d, dt.time()) for d in dates]
        ax.plot(x_vals, interpolated, color=color, linewidth=2, label=label)
        ax.plot(x_vals, average, color="#111827", linestyle="--", label="7-day average")
        ax.set_title(label)
        ax.legend()
        fig.autofmt_xdate()
        fig.canvas.draw()

    def plot_maintenance_range(self, dates, low, high, mid):
        self.maintenance_ax.clear()
        self.maintenance_ax.grid(True, linestyle="--", alpha=0.3)
        if not dates:
            self.maintenance_ax.set_title("No data yet")
            self.maintenance_fig.canvas.draw()
            return

        x_vals = [dt.datetime.combine(d, dt.time()) for d in dates]
        self.maintenance_ax.fill_between(
            x_vals, low, high, color="#fca5a5", alpha=0.4, label="Estimated range"
        )
        self.maintenance_ax.plot(
            x_vals, mid, color="#991b1b", linewidth=2, label="Mid estimate"
        )
        self.maintenance_ax.set_title("Maintenance calories")
        self.maintenance_ax.legend()
        self.maintenance_fig.autofmt_xdate()
        self.maintenance_fig.canvas.draw()

    def on_select_row(self, _event):
        selection = self.tree.selection()
        if not selection:
            return
        values = self.tree.item(selection[0], "values")
        self.date_var.set(values[0])
        self.weight_var.set(values[1])
        self.waist_var.set(values[2])
        self.calories_var.set(values[3])
        self.calories_accuracy_var.set(values[4])

    def set_today(self):
        self.date_var.set(format_date(dt.date.today()))

    def clear_form(self):
        self.weight_var.set("")
        self.waist_var.set("")
        self.calories_var.set("")
        self.calories_accuracy_var.set("")

    def save_entry(self):
        date_str = self.date_var.get().strip()
        if not date_str:
            messagebox.showerror("Missing date", "Please enter a date.")
            return
        try:
            parse_date(date_str)
        except ValueError:
            messagebox.showerror("Invalid date", "Use format YYYY-MM-DD.")
            return

        weight, weight_ok = self.parse_float(self.weight_var.get().strip(), "weight")
        waist, waist_ok = self.parse_float(self.waist_var.get().strip(), "waist")
        calories, calories_ok = self.parse_float(self.calories_var.get().strip(), "calories")
        accuracy, accuracy_ok = self.parse_float(
            self.calories_accuracy_var.get().strip(), "calories accuracy"
        )

        if not weight_ok or not waist_ok or not calories_ok or not accuracy_ok:
            return

        if calories is not None and calories < 0:
            messagebox.showerror("Invalid value", "Calories must be positive.")
            return
        if accuracy is not None and accuracy < 0:
            messagebox.showerror("Invalid value", "Calories accuracy must be positive.")
            return

        if calories is None and accuracy is not None:
            messagebox.showerror("Missing calories", "Enter calories before accuracy.")
            return

        if calories is not None and accuracy is None:
            accuracy = 0.0

        if weight is None and waist is None and calories is None:
            messagebox.showerror(
                "Missing data", "Enter weight, waist, and/or calorie data."
            )
            return

        upsert_measurement(self.conn, date_str, weight, waist, calories, accuracy)
        self.refresh_table()
        self.refresh_charts()

    def delete_entry(self):
        date_str = self.date_var.get().strip()
        if not date_str:
            messagebox.showerror("Missing date", "Please enter a date.")
            return
        try:
            parse_date(date_str)
        except ValueError:
            messagebox.showerror("Invalid date", "Use format YYYY-MM-DD.")
            return

        delete_measurement(self.conn, date_str)
        self.refresh_table()
        self.refresh_charts()
        self.clear_form()

    def save_profile(self):
        height, height_ok = self.parse_float(self.height_var.get().strip(), "height")
        age, age_ok = self.parse_int(self.age_var.get().strip(), "age")
        baseline_weight, weight_ok = self.parse_float(
            self.baseline_weight_var.get().strip(), "baseline weight"
        )
        baseline_maintenance, maintenance_ok = self.parse_float(
            self.baseline_maintenance_var.get().strip(), "baseline maintenance"
        )

        if not (height_ok and age_ok and weight_ok and maintenance_ok):
            return

        if height is not None and height <= 0:
            messagebox.showerror("Invalid value", "Height must be positive.")
            return
        if age is not None and age <= 0:
            messagebox.showerror("Invalid value", "Age must be positive.")
            return
        if baseline_weight is not None and baseline_weight <= 0:
            messagebox.showerror("Invalid value", "Baseline weight must be positive.")
            return
        if baseline_maintenance is not None and baseline_maintenance <= 0:
            messagebox.showerror("Invalid value", "Baseline maintenance must be positive.")
            return

        self.profile = {
            "height_cm": height,
            "age_years": age,
            "sex": self.sex_var.get(),
            "activity_level": self.activity_var.get(),
            "baseline_weight": baseline_weight,
            "baseline_maintenance": baseline_maintenance or MAINTENANCE_BASELINE_DEFAULT,
        }
        save_profile(self.conn, self.profile)
        self.refresh_charts()

    def parse_float(self, value, label):
        if not value:
            return None, True
        try:
            return float(value), True
        except ValueError:
            messagebox.showerror("Invalid value", f"Enter a valid number for {label}.")
            return None, False

    def parse_int(self, value, label):
        if not value:
            return None, True
        try:
            return int(value), True
        except ValueError:
            messagebox.showerror("Invalid value", f"Enter a valid integer for {label}.")
            return None, False


if __name__ == "__main__":
    app = WeightTrackerApp()
    app.mainloop()
