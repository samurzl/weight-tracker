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
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
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


def get_setting(conn, key, default=None):
    cur = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    if row:
        return row[0]
    return default


def set_setting(conn, key, value):
    conn.execute(
        """
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
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


def fill_calories_series(values, window=7):
    if not values:
        return []

    def is_missing(value):
        return value is None or (isinstance(value, float) and math.isnan(value))

    filled = []
    total = len(values)
    for index, value in enumerate(values):
        if not is_missing(value):
            filled.append(float(value))
            continue

        backward = []
        for prev_index in range(index - 1, -1, -1):
            candidate = values[prev_index]
            if is_missing(candidate):
                break
            backward.append(float(candidate))
            if len(backward) == window:
                break

        forward = []
        for next_index in range(index + 1, total):
            candidate = values[next_index]
            if is_missing(candidate):
                break
            forward.append(float(candidate))
            if len(forward) == window:
                break

        if len(backward) >= window:
            filled.append(float(np.mean(backward[:window])))
        elif len(forward) >= window:
            filled.append(float(np.mean(forward[:window])))
        elif backward or forward:
            chosen = backward if len(backward) >= len(forward) else forward
            filled.append(float(np.mean(chosen)))
        else:
            filled.append(np.nan)

    return filled


def moving_average(values, window=7):
    if len(values) < window:
        return [np.nan for _ in values]

    kernel = np.ones(window) / window
    conv = np.convolve(values, kernel, mode="valid")
    prefix = [np.nan] * (window - 1)
    return prefix + conv.tolist()


def calculate_body_fat_percent(weight_kg, waist_cm, sex):
    if weight_kg is None or waist_cm is None:
        return np.nan
    if isinstance(weight_kg, float) and math.isnan(weight_kg):
        return np.nan
    if isinstance(waist_cm, float) and math.isnan(waist_cm):
        return np.nan

    weight_lb = weight_kg * 2.2046226218
    waist_in = waist_cm / 2.54
    if sex == "female":
        body_fat = ((4.15 * waist_in) - (0.082 * weight_lb) - 76.76) / weight_lb * 100
    else:
        body_fat = ((4.15 * waist_in) - (0.082 * weight_lb) - 98.42) / weight_lb * 100
    return body_fat


def calculate_body_fat_mass(weight_kg, waist_cm, sex):
    body_fat_percent = calculate_body_fat_percent(weight_kg, waist_cm, sex)
    if math.isnan(body_fat_percent):
        return np.nan
    return weight_kg * (body_fat_percent / 100)


def calculate_body_fat_mass_series(weight_values, waist_values, sex):
    if not weight_values:
        return []
    weight_interp = interpolate_series(weight_values)
    waist_interp = interpolate_series(waist_values)
    return [
        calculate_body_fat_mass(weight, waist, sex)
        for weight, waist in zip(weight_interp, waist_interp)
    ]


def calculate_maintenance_range(
    dates,
    calories,
    calories_accuracy,
    weight_values,
    waist_values,
    sex,
):
    if not dates:
        return [], [], [], []

    weight_interp = interpolate_series(weight_values)
    waist_interp = interpolate_series(waist_values)
    fat_mass = calculate_body_fat_mass_series(weight_interp, waist_interp, sex)
    calories_interp = fill_calories_series(calories)
    accuracy_interp = interpolate_series(calories_accuracy)

    maintenance_low = []
    maintenance_high = []
    maintenance_mid = []
    maintenance_error = []

    total_days = len(dates)
    baseline = float(np.nanmean(calories_interp)) if calories_interp else 2000.0

    fat_kcal_per_kg = 7700.0
    learning_rate = 0.15
    window = 7

    def window_sum(values, start_index, end_index):
        window_values = values[start_index:end_index]
        if any(math.isnan(v) for v in window_values):
            return np.nan
        return float(np.sum(window_values))

    for index in range(total_days):
        lookback = min(window, index)
        if lookback < 1:
            maintenance_low.append(baseline)
            maintenance_high.append(baseline)
            maintenance_mid.append(baseline)
            maintenance_error.append(np.nan)
            continue

        prior_start = index - lookback
        prior_end = index
        prior_calories_sum = window_sum(calories_interp, prior_start, prior_end)
        prior_accuracy = [
            0.0 if math.isnan(v) else v for v in accuracy_interp[prior_start:prior_end]
        ]
        prior_low_sum = (
            prior_calories_sum - float(np.sum(prior_accuracy))
            if not math.isnan(prior_calories_sum)
            else np.nan
        )
        prior_high_sum = (
            prior_calories_sum + float(np.sum(prior_accuracy))
            if not math.isnan(prior_calories_sum)
            else np.nan
        )

        fat_today = fat_mass[index - 1]
        fat_then = fat_mass[index - lookback]
        if math.isnan(prior_calories_sum) or math.isnan(fat_today) or math.isnan(fat_then):
            estimate_mid = baseline
            estimate_low = baseline
            estimate_high = baseline
        else:
            fat_change = fat_today - fat_then
            estimate_mid = (prior_calories_sum - fat_change * fat_kcal_per_kg) / lookback
            estimate_low = (
                (prior_low_sum - fat_change * fat_kcal_per_kg) / lookback
                if not math.isnan(prior_low_sum)
                else estimate_mid
            )
            estimate_high = (
                (prior_high_sum - fat_change * fat_kcal_per_kg) / lookback
                if not math.isnan(prior_high_sum)
                else estimate_mid
            )

        maintenance_low.append(estimate_low)
        maintenance_high.append(estimate_high)
        maintenance_mid.append(estimate_mid)

        forward_len = min(window, total_days - index)
        if forward_len < 1 or index + forward_len - 1 >= total_days:
            maintenance_error.append(np.nan)
            continue

        next_calories_sum = window_sum(calories_interp, index, index + forward_len)
        if (
            math.isnan(next_calories_sum)
            or math.isnan(estimate_mid)
            or math.isnan(fat_today)
            or math.isnan(fat_mass[index + forward_len - 1])
        ):
            maintenance_error.append(np.nan)
            continue

        predicted_change = (
            next_calories_sum - estimate_mid * forward_len
        ) / fat_kcal_per_kg
        actual_change = fat_mass[index + forward_len - 1] - fat_today
        error_kg = actual_change - predicted_change
        maintenance_error.append(error_kg * fat_kcal_per_kg)

        deficit = estimate_mid * forward_len - next_calories_sum
        if actual_change != 0 and not math.isnan(deficit):
            implied_kcal = deficit / (-actual_change)
            if implied_kcal > 0 and math.isfinite(implied_kcal):
                fat_kcal_per_kg += (implied_kcal - fat_kcal_per_kg) * learning_rate

    return maintenance_low, maintenance_high, maintenance_mid, maintenance_error


class WeightTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Weight Tracker")
        self.geometry("1100x700")
        self.minsize(1000, 650)

        self.conn = ensure_db()
        self.sex = get_setting(self.conn, "sex", "male")

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
                "Body fat mass uses waist and weight (YMCA formula). "
                "Maintenance calories are tuned using body fat changes."
            ),
            foreground="#555555",
        )
        note.grid(row=3, column=0, columnspan=5, sticky=tk.W, pady=(10, 0))

        sex_frame = ttk.Frame(form_frame)
        sex_frame.grid(row=4, column=0, columnspan=5, sticky=tk.W, pady=(6, 0))
        ttk.Label(sex_frame, text="Sex for body fat mass").pack(side=tk.LEFT)
        self.sex_var = tk.StringVar(value=self.sex)
        sex_combo = ttk.Combobox(
            sex_frame, textvariable=self.sex_var, values=["male", "female"], width=8
        )
        sex_combo.pack(side=tk.LEFT, padx=(8, 0))
        sex_combo.state(["readonly"])
        sex_combo.bind("<<ComboboxSelected>>", self.on_sex_change)

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
        self.maintenance_error_tab = ttk.Frame(notebook)

        notebook.add(self.weight_tab, text="Weight")
        notebook.add(self.waist_tab, text="Waist")
        notebook.add(self.bodyfat_tab, text="Body fat index")
        notebook.add(self.calories_tab, text="Calories")
        notebook.add(self.maintenance_tab, text="Maintenance range")
        notebook.add(self.maintenance_error_tab, text="Maintenance error")

        self.weight_fig, self.weight_ax = self.create_chart(self.weight_tab)
        self.waist_fig, self.waist_ax = self.create_chart(self.waist_tab)
        self.bodyfat_fig, self.bodyfat_ax = self.create_chart(self.bodyfat_tab)
        self.calories_fig, self.calories_ax = self.create_chart(self.calories_tab)
        self.maintenance_fig, self.maintenance_ax = self.create_chart(self.maintenance_tab)
        self.maintenance_error_fig, self.maintenance_error_ax = self.create_chart(
            self.maintenance_error_tab
        )

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
        bodyfat = (
            calculate_body_fat_mass_series(weight_values, waist_values, self.sex)
            if body_dates
            else []
        )

        self.plot_metric(
            self.bodyfat_ax,
            self.bodyfat_fig,
            body_dates,
            bodyfat,
            "Body fat mass (kg)",
            color="#f97316",
        )

        self.plot_metric(
            self.calories_ax,
            self.calories_fig,
            calories_dates,
            calories_values,
            "Calories",
            color="#9333ea",
            fill_strategy=fill_calories_series,
        )

        maintenance_dates = weight_dates if weight_dates else calories_dates
        low, high, mid, error = calculate_maintenance_range(
            maintenance_dates,
            calories_values,
            accuracy_values,
            weight_values,
            waist_values,
            self.sex,
        )
        self.plot_maintenance_range(maintenance_dates, low, high, mid)
        self.plot_maintenance_error(maintenance_dates, error)

    def plot_metric(self, ax, fig, dates, values, label, color, fill_strategy=interpolate_series):
        ax.clear()
        ax.grid(True, linestyle="--", alpha=0.3)
        if not dates:
            ax.set_title("No data yet")
            fig.canvas.draw()
            return

        interpolated = fill_strategy(values)
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

    def plot_maintenance_error(self, dates, error_values):
        self.maintenance_error_ax.clear()
        self.maintenance_error_ax.grid(True, linestyle="--", alpha=0.3)
        if not dates:
            self.maintenance_error_ax.set_title("No data yet")
            self.maintenance_error_fig.canvas.draw()
            return

        x_vals = [dt.datetime.combine(d, dt.time()) for d in dates]
        self.maintenance_error_ax.plot(
            x_vals, error_values, color="#dc2626", linewidth=2, label="Prediction error"
        )
        self.maintenance_error_ax.axhline(0, color="#111827", linewidth=1)
        self.maintenance_error_ax.set_title("Maintenance prediction error (kcal)")
        self.maintenance_error_ax.legend()
        self.maintenance_error_fig.autofmt_xdate()
        self.maintenance_error_fig.canvas.draw()

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

    def on_sex_change(self, _event):
        self.sex = self.sex_var.get()
        set_setting(self.conn, "sex", self.sex)
        self.refresh_charts()

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

    def parse_float(self, value, label):
        if not value:
            return None, True
        try:
            return float(value), True
        except ValueError:
            messagebox.showerror("Invalid value", f"Enter a valid number for {label}.")
            return None, False


if __name__ == "__main__":
    app = WeightTrackerApp()
    app.mainloop()
