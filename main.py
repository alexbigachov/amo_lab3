"""
Лабораторна робота №3
Тема: Інтерполяція функцій. Інтерполяційні многочлени.
Варіант 1: f(x) = sin(x^2), [0, 2], формула Лагранжа
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ────────────────────
#  Математична частина
# ────────────────────

def f_target(x):
    """Задана функція варіанту 1"""
    return math.sin(x ** 2)

def f_test(x):
    """Тестова функція sin(x) для налагодження"""
    return math.sin(x)

def build_nodes(a, b, n):
    """Рівновіддалені вузли від a до b, n+1 вузол"""
    h = (b - a) / n
    xs = [a + i * h for i in range(n + 1)]
    return xs

def lagrange(xs, ys, x):
    """Інтерполяційний многочлен Лагранжа"""
    n = len(xs)
    result = 0.0
    for i in range(n):
        L = 1.0
        for j in range(n):
            if j != i:
                denom = xs[i] - xs[j]
                if abs(denom) < 1e-15:
                    return float('nan')
                L *= (x - xs[j]) / denom
        result += ys[i] * L
    return result

def newton_divided_diff(xs, ys):
    """Таблиця розділених різниць для Ньютона"""
    n = len(xs)
    dd = [list(ys)]
    for k in range(1, n):
        prev = dd[k - 1]
        curr = []
        for i in range(n - k):
            denom = xs[i + k] - xs[i]
            if abs(denom) < 1e-15:
                curr.append(0.0)
            else:
                curr.append((prev[i + 1] - prev[i]) / denom)
        dd.append(curr)
    return dd

def newton(xs, ys, x):
    """Інтерполяційний многочлен Ньютона"""
    dd = newton_divided_diff(xs, ys)
    n = len(xs)
    result = dd[0][0]
    product = 1.0
    for k in range(1, n):
        product *= (x - xs[k - 1])
        result += dd[k][0] * product
    return result

def aitken(xs, ys, x):
    """Рекурентне співвідношення Ейткена"""
    n = len(xs)
    Q = list(ys)
    for k in range(1, n):
        new_Q = []
        for i in range(n - k):
            denom = xs[i + k] - xs[i]
            if abs(denom) < 1e-15:
                new_Q.append(Q[i])
            else:
                val = ((x - xs[i]) * Q[i + 1] - (x - xs[i + k]) * Q[i]) / denom
                new_Q.append(val)
        Q = new_Q
    return Q[0]

def estimate_error(xs, ys_n, ys_n1, x_vals):
    """
    Оцінка похибки: порівняння значень многочленів степені n і n+1.
    Повертає список |P{n+1}(x) - Pn(x)| для кожного x у x_vals.
    """
    errors = []
    for x in x_vals:
        pn = lagrange(xs[:-1], ys_n, x)
        pn1 = lagrange(xs, ys_n1, x)
        errors.append(abs(pn1 - pn))
    return errors


# ───────
#  GUI
# ───────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лаб. робота №3 — Інтерполяція функцій")
        self.configure(bg="#0f1117")
        self.resizable(True, True)
        self.geometry("1200x820")
        self._build_ui()

    # ── layout ──────────────────────────────
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0f1117")
        style.configure("TLabel", background="#0f1117", foreground="#e2e8f0",
                        font=("Consolas", 10))
        style.configure("Head.TLabel", background="#0f1117", foreground="#60a5fa",
                        font=("Consolas", 12, "bold"))
        style.configure("TButton", background="#1e40af", foreground="#e2e8f0",
                        font=("Consolas", 10, "bold"), relief="flat", padding=6)
        style.map("TButton",
                  background=[("active", "#2563eb")],
                  foreground=[("active", "#ffffff")])
        style.configure("TCombobox", fieldbackground="#1e293b", background="#1e293b",
                        foreground="#e2e8f0", font=("Consolas", 10))
        style.configure("TEntry", fieldbackground="#1e293b", foreground="#e2e8f0",
                        font=("Consolas", 10))
        style.configure("TNotebook", background="#0f1117", tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab", background="#1e293b", foreground="#94a3b8",
                        font=("Consolas", 10), padding=[12, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", "#1e40af")],
                  foreground=[("selected", "#ffffff")])

        # ── top bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=16, pady=(14, 6))
        ttk.Label(top, text="⟨ Інтерполяція функцій ⟩", style="Head.TLabel").pack(side="left")
        ttk.Label(top, text="Варіант 1 · f(x) = sin(x²) · [0, 2] · формула Лагранжа",
                  foreground="#64748b", font=("Consolas", 9)).pack(side="left", padx=16)

        # ── main panes
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=16, pady=6)

        # left panel – controls
        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)
        self._build_controls(left)

        # right panel – notebook
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self._build_notebook(right)

    def _build_controls(self, parent):
        parent.configure(padding=(0, 0, 12, 0))

        def row(lbl, widget_factory):
            fr = ttk.Frame(parent)
            fr.pack(fill="x", pady=3)
            ttk.Label(fr, text=lbl, width=18, anchor="e").pack(side="left")
            w = widget_factory(fr)
            w.pack(side="left", fill="x", expand=True, padx=(6, 0))
            return w

        # function selector
        ttk.Label(parent, text="── Налаштування ──", style="Head.TLabel").pack(anchor="w", pady=(4, 8))
        self.func_var = tk.StringVar(value="Варіант: sin(x²)")
        row("Функція:", lambda p: ttk.Combobox(p, textvariable=self.func_var,
            values=["Варіант: sin(x²)", "Тест: sin(x)"], state="readonly"))

        self.a_var = tk.StringVar(value="0")
        self.b_var = tk.StringVar(value="2")
        self.n_var = tk.StringVar(value="10")
        self.method_var = tk.StringVar(value="Лагранж")
        self.xq_var = tk.StringVar(value="0.75")

        row("a =", lambda p: ttk.Entry(p, textvariable=self.a_var))
        row("b =", lambda p: ttk.Entry(p, textvariable=self.b_var))
        row("n (вузлів-1):", lambda p: ttk.Entry(p, textvariable=self.n_var))
        row("Метод:", lambda p: ttk.Combobox(p, textvariable=self.method_var,
            values=["Лагранж", "Ньютон", "Ейткен"], state="readonly"))
        row("x запиту:", lambda p: ttk.Entry(p, textvariable=self.xq_var))

        ttk.Button(parent, text="▶  Обчислити", command=self._run).pack(fill="x", pady=(16, 4))
        ttk.Button(parent, text="↺  Скинути", command=self._reset).pack(fill="x", pady=4)

        # results box
        ttk.Label(parent, text="── Результат ──", style="Head.TLabel").pack(anchor="w", pady=(18, 4))
        self.result_text = tk.Text(parent, height=18, width=34, bg="#1e293b", fg="#a5f3fc",
                                   font=("Consolas", 9), relief="flat", insertbackground="#e2e8f0",
                                   state="disabled", wrap="word")
        self.result_text.pack(fill="both", expand=True)

    def _build_notebook(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        self.tab_graph = ttk.Frame(nb)
        self.tab_error = ttk.Frame(nb)
        self.tab_table = ttk.Frame(nb)
        self.tab_bloksch = ttk.Frame(nb)

        nb.add(self.tab_graph, text="  Графік інтерполяції  ")
        nb.add(self.tab_error, text="  Графік похибки  ")
        nb.add(self.tab_table, text="  Таблиця значень  ")

        self._init_graph_tab()
        self._init_error_tab()
        self._init_table_tab()

    # ── tabs init ────────────────────────────
    def _init_graph_tab(self):
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0f1117")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.set_title("Інтерполяція", color="#60a5fa", fontsize=11)
        self.fig_graph = fig
        self.ax_graph = ax
        canvas = FigureCanvasTkAgg(fig, master=self.tab_graph)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_graph = canvas

    def _init_error_tab(self):
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0f1117")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.set_title("Оцінка похибки -lg(Δ)", color="#60a5fa", fontsize=11)
        self.fig_err = fig
        self.ax_err = ax
        canvas = FigureCanvasTkAgg(fig, master=self.tab_error)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_err = canvas

    def _init_table_tab(self):
        cols = ("i", "xᵢ", "f(xᵢ)", "Pₙ(xᵢ)", "Δᵢ = |f − Pₙ|")
        tv = ttk.Treeview(self.tab_table, columns=cols, show="headings", height=22)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=150, anchor="center")
        tv.tag_configure("even", background="#1e293b", foreground="#e2e8f0")
        tv.tag_configure("odd", background="#0f172a", foreground="#e2e8f0")

        style = ttk.Style()
        style.configure("Treeview", background="#1e293b", fieldbackground="#1e293b",
                        foreground="#e2e8f0", font=("Consolas", 9), rowheight=22)
        style.configure("Treeview.Heading", background="#1e40af", foreground="#ffffff",
                        font=("Consolas", 9, "bold"))

        sb = ttk.Scrollbar(self.tab_table, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.table_tv = tv

    # ── logic ────────────────────────────────
    def _get_func(self):
        if "sin(x²)" in self.func_var.get():
            return f_target, "sin(x²)"
        return f_test, "sin(x)"

    def _run(self):
        try:
            a = float(self.a_var.get())
            b = float(self.b_var.get())
            n = int(self.n_var.get())
            method = self.method_var.get()
            xq = float(self.xq_var.get())
            assert a < b and n >= 1
        except Exception:
            messagebox.showerror("Помилка", "Перевірте вхідні параметри.")
            return

        func, fname = self._get_func()
        xs = build_nodes(a, b, n)
        ys = [func(x) for x in xs]

        # Вибір методу
        if method == "Лагранж":
            interp = lambda xv: lagrange(xs, ys, xv)
        elif method == "Ньютон":
            interp = lambda xv: newton(xs, ys, xv)
        else:
            interp = lambda xv: aitken(xs, ys, xv)

        # Значення у вузлах
        rows = []
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            pi = interp(xi)
            rows.append((i, xi, yi, pi, abs(yi - pi)))

        # Таблиця
        self._fill_table(rows)

        # Точка запиту
        p_xq = interp(xq)
        true_xq = func(xq)
        err_xq = abs(true_xq - p_xq)

        # Графік інтерполяції
        x_plot = [a + (b - a) * t / 500 for t in range(501)]
        y_true = [func(x) for x in x_plot]
        y_interp = [interp(x) for x in x_plot]

        ax = self.ax_graph
        ax.clear()
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.plot(x_plot, y_true, color="#34d399", linewidth=2, label=f"f(x) = {fname}")
        ax.plot(x_plot, y_interp, color="#f472b6", linewidth=1.5, linestyle="--", label=f"P_{n}(x) ({method})")
        ax.scatter(xs, ys, color="#facc15", s=40, zorder=5, label="Вузли")
        ax.axvline(xq, color="#a78bfa", linestyle=":", linewidth=1)
        ax.scatter([xq], [p_xq], color="#a78bfa", s=60, zorder=6, label=f"P({xq:.3f})={p_xq:.6f}")
        ax.set_title(f"Інтерполяція: {fname}, n={n}, метод={method}", color="#60a5fa", fontsize=10)
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
        ax.grid(True, color="#1e3a5f", linewidth=0.5)
        self.fig_graph.tight_layout()
        self.canvas_graph.draw()

        # Графік похибки для різних n
        ax2 = self.ax_err
        ax2.clear()
        ax2.set_facecolor("#1e293b")
        ax2.tick_params(colors="#94a3b8")
        for sp in ax2.spines.values(): sp.set_color("#334155")

        x_eval = [a + (b - a) * t / 200 for t in range(201)]
        x_norm = [(x - a) / (b - a) for x in x_eval]
        colors_e = plt.cm.cool(np.linspace(0, 1, min(n, 9)))

        for k in range(1, min(n + 1, 10)):
            xs_k = build_nodes(a, b, k)
            ys_k = [func(x) for x in xs_k]
            # похибка = |f(x) - Pₙ(x)| в кожній точці
            errs = [abs(func(x) - lagrange(xs_k, ys_k, x)) for x in x_eval]
            # замінюємо нулі щоб уникнути log(0)
            log_errs = [-math.log10(e) if e > 1e-10 else float('nan') for e in errs]
            ax2.plot(x_norm, log_errs,
                     color=colors_e[k - 1], linewidth=1.2, label=f"n={k}")

        ax2.set_xlabel("x̄ = (x−a)/(b−a)", color="#94a3b8")
        ax2.set_ylabel("−lg(Δₙ)", color="#94a3b8")
        ax2.set_title("Оцінка похибки −lg(Δₙ) при різних ступенях n", color="#60a5fa", fontsize=10)
        ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=7, ncol=3)
        ax2.grid(True, color="#1e3a5f", linewidth=0.5)
        self.fig_err.tight_layout()
        self.canvas_err.draw()

        # Текст результатів
        self._set_result(
            f"Функція : {fname}\n"
            f"Відрізок: [{a}, {b}]\n"
            f"Вузлів  : n+1 = {n+1}\n"
            f"Метод   : {method}\n"
            "─────────────────────\n"
            f"x*      = {xq}\n"
            f"f(x*)   = {true_xq:.10f}\n"
            f"Pₙ(x*)  = {p_xq:.10f}\n"
            f"Похибка = {err_xq:.3e}\n"
            "─────────────────────\n"
            f"Max|Δ| у вузлах:\n"
            f"  {max(r[4] for r in rows):.3e}\n"
        )

    def _fill_table(self, rows):
        tv = self.table_tv
        for item in tv.get_children():
            tv.delete(item)
        for r in rows:
            tag = "even" if r[0] % 2 == 0 else "odd"
            tv.insert("", "end", values=(
                r[0],
                f"{r[1]:.6f}",
                f"{r[2]:.8f}",
                f"{r[3]:.8f}",
                f"{r[4]:.3e}"
            ), tags=(tag,))

    def _set_result(self, txt):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", txt)
        self.result_text.configure(state="disabled")

    def _reset(self):
        self.a_var.set("0")
        self.b_var.set("2")
        self.n_var.set("10")
        self.method_var.set("Лагранж")
        self.xq_var.set("0.75")
        self.func_var.set("Варіант: sin(x²)")
        self._set_result("")
        for item in self.table_tv.get_children():
            self.table_tv.delete(item)


if __name__ == "__main__":
    app = App()
    app.mainloop()