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


matplotlib.use("TkAgg")


def ensure_db():
    os.makedirs(APP_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS measurements (
            date TEXT PRIMARY KEY,
            weight REAL,
            waist REAL
        )
        """
    )
    conn.commit()
    return conn


def upsert_measurement(conn, date_str, weight, waist):
    conn.execute(
        """
        INSERT INTO measurements (date, weight, waist)
        VALUES (?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET weight = excluded.weight, waist = excluded.waist
        """,
        (date_str, weight, waist),
    )
    conn.commit()


def delete_measurement(conn, date_str):
    conn.execute("DELETE FROM measurements WHERE date = ?", (date_str,))
    conn.commit()


def fetch_measurements(conn):
    cur = conn.execute("SELECT date, weight, waist FROM measurements ORDER BY date")
    rows = cur.fetchall()
    return rows


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


class WeightTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Weight Tracker")
        self.geometry("1100x700")
        self.minsize(1000, 650)

        self.conn = ensure_db()

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

        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

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
            text="Body fat index is calculated as waist / weight. Lower is leaner.",
            foreground="#555555",
        )
        note.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

        self.tree = ttk.Treeview(
            table_frame,
            columns=("date", "weight", "waist"),
            show="headings",
            height=8,
        )
        self.tree.heading("date", text="Date")
        self.tree.heading("weight", text="Weight (kg)")
        self.tree.heading("waist", text="Waist (cm)")
        self.tree.column("date", width=110)
        self.tree.column("weight", width=110)
        self.tree.column("waist", width=110)
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

        notebook.add(self.weight_tab, text="Weight")
        notebook.add(self.waist_tab, text="Waist")
        notebook.add(self.bodyfat_tab, text="Body fat index")

        self.weight_fig, self.weight_ax = self.create_chart(self.weight_tab)
        self.waist_fig, self.waist_ax = self.create_chart(self.waist_tab)
        self.bodyfat_fig, self.bodyfat_ax = self.create_chart(self.bodyfat_tab)

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
        for date_str, weight, waist in rows:
            weight_text = f"{weight:.1f}" if weight is not None else ""
            waist_text = f"{waist:.1f}" if waist is not None else ""
            self.tree.insert("", tk.END, values=(date_str, weight_text, waist_text))

    def refresh_charts(self):
        rows = fetch_measurements(self.conn)
        weight_dates, weight_values = build_series(rows, 1)
        waist_dates, waist_values = build_series(rows, 2)

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

    def on_select_row(self, _event):
        selection = self.tree.selection()
        if not selection:
            return
        values = self.tree.item(selection[0], "values")
        self.date_var.set(values[0])
        self.weight_var.set(values[1])
        self.waist_var.set(values[2])

    def set_today(self):
        self.date_var.set(format_date(dt.date.today()))

    def clear_form(self):
        self.weight_var.set("")
        self.waist_var.set("")

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

        if not weight_ok or not waist_ok:
            return

        if weight is None and waist is None:
            messagebox.showerror("Missing data", "Enter a weight and/or waist measurement.")
            return

        upsert_measurement(self.conn, date_str, weight, waist)
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
