"""سجل التدريب بصيغة CSV، ورسم منحنياته بـ matplotlib.

CSV نص بسيط يُفتح في أي برنامج جداول، ويُكمَل بعد الاستئناف دون أن يضيع ما قبله.
والرسم يُحفظ صورةً، فيعمل على Colab وعلى خادم بلا شاشة.

عناوين الرسوم بالإنجليزية لأن matplotlib لا يصل الحروف العربية ولا يعكس اتجاهها.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


class CSVLogger:
    """يكتب صفًا لكل خطوة. الأعمدة تُحدَّد مرة واحدة، والملف الموجود يُكمَل ولا يُمسح."""

    def __init__(self, path: str | Path, fields: Iterable[str]):
        self.path = Path(path)
        self.fields = list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != self.fields:
                raise ValueError(
                    f"existing log {self.path} has columns {header}, expected {self.fields}"
                )
        else:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.fields)

    def log(self, row: Mapping[str, object]) -> None:
        """إضافة صف. العمود المجهول خطأ صريح، حتى لا يضيع رقم بسبب خطأ إملائي."""
        unknown = sorted(set(row) - set(self.fields))
        if unknown:
            raise KeyError(f"unknown log columns: {unknown}")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(field, "") for field in self.fields])


def read_log(path: str | Path) -> list[dict[str, str]]:
    """قراءة السجل كقائمة من القواميس، صف لكل خطوة."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_curves(
    path: str | Path, out: str | Path, x: str, ys: Iterable[str], title: str = ""
) -> Path:
    """رسم الأعمدة ``ys`` مقابل العمود ``x`` من سجل CSV، وحفظ الصورة في ``out``."""
    import matplotlib

    matplotlib.use("Agg")  # بلا شاشة: نحفظ صورة فقط
    import matplotlib.pyplot as plt

    rows = read_log(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for y in ys:
        points = [(float(r[x]), float(r[y])) for r in rows if r.get(y) not in (None, "")]
        if points:
            xs, values = zip(*points)
            ax.plot(xs, values, label=y)
    ax.set_xlabel(x)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    if title:
        ax.set_title(title)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
