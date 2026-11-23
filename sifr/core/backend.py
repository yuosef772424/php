"""اختيار الواجهة الخلفية للحساب: NumPy على المعالج أو CuPy على GPU.

كل كود النموذج يكتب ``xp.exp`` و``xp.matmul`` بدل ``numpy.exp``، و``xp`` هنا هو
NumPy أو CuPy. لماذا: الكود نفسه يعمل على المعالج (للتحقق بدقة float64) وعلى
GPU (للتدريب بدقة float32)، فلا نكتب النموذج مرتين ولا تختلف النسختان في المنطق.

الفروق القليلة بين المكتبتين (مثل الجمع المبعثر) نخفيها هنا، فلا يعرف بقية
الكود أي مكتبة يستعمل.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def gpu_available() -> bool:
    """هل توجد مكتبة CuPy وكرت GPU واحد على الأقل؟"""
    try:
        import cupy  # مكتبة اختيارية: غيابها يعني العمل على المعالج، لا خطأ

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # لا CuPy، أو لا تعريف CUDA، أو لا كرت
        return False


@dataclass(frozen=True)
class Backend:
    """واجهة خلفية واحدة: اسمها، ووحدة المصفوفات ``xp``، ونوع الأعداد الافتراضي."""

    name: str  # "numpy" أو "cupy"
    xp: Any
    dtype: Any

    @property
    def is_gpu(self) -> bool:
        return self.name == "cupy"

    def asarray(self, data, dtype=None):
        """تحويل أي بيانات إلى مصفوفة على هذه الواجهة، بالنوع الافتراضي إن لم يُحدَّد غيره."""
        return self.xp.asarray(data, dtype=dtype or self.dtype)

    def to_numpy(self, array) -> np.ndarray:
        """إعادة المصفوفة إلى ذاكرة المعالج، للحفظ والطباعة والمقارنة."""
        if self.is_gpu:
            return self.xp.asnumpy(array)
        return np.asarray(array)

    def scatter_add(self, target, indices, values) -> None:
        """جمع ``values`` في صفوف ``target`` المحددة بـ ``indices``، مع جمع المكرر.

        نحتاجه في الانتشار العكسي لجدول التضمين: التوكن الذي يظهر مرتين في
        الدفعة يجب أن يُجمع تدرجه مرتين. الكتابة المباشرة
        ``target[indices] += values`` تحتفظ بآخر قيمة فقط للمواضع المكررة، فتضيع
        معلومة بصمت؛ لذلك لا نستعملها.
        """
        if self.is_gpu:
            import cupyx

            cupyx.scatter_add(target, indices, values)
        else:
            np.add.at(target, indices, values)


def get_backend(name: str = "auto", dtype: str | None = None) -> Backend:
    """إرجاع الواجهة الخلفية المطلوبة.

    - ``"cpu"`` أو ``"numpy"``: NumPy، والنوع الافتراضي float64 (للتحقق).
    - ``"gpu"`` أو ``"cupy"``: CuPy، والنوع الافتراضي float32 (للتدريب).
      إن لم يوجد GPU فهذا خطأ صريح، لا تحويل صامت إلى المعالج.
    - ``"auto"``: CuPy إن وُجد GPU، وإلا NumPy.
    """
    key = name.lower()
    if key == "auto":
        key = "gpu" if gpu_available() else "cpu"
    if key in ("cpu", "numpy"):
        return Backend("numpy", np, np.dtype(dtype or "float64"))
    if key in ("gpu", "cupy"):
        if not gpu_available():
            raise RuntimeError("backend 'cupy' requested, but CuPy or a GPU is not available here")
        import cupy

        return Backend("cupy", cupy, cupy.dtype(dtype or "float32"))
    raise ValueError(f"unknown backend {name!r} (expected: auto, cpu, numpy, gpu, cupy)")
