"""البذور العشوائية: مصدر واحد معلن لكل ما هو عشوائي في المشروع.

نفس البذرة تعيد التجربة نفسها حرفيًا على المعالج. ونشتق من البذرة الرئيسية بذرة
مستقلة لكل غرض (تهيئة الأوزان، خلط البيانات، dropout). لماذا: لو تشاركت الأغراض
مولّدًا واحدًا، لتغيّر ترتيب البيانات لمجرد أننا غيّرنا حجم النموذج، لأن التهيئة
استهلكت أرقامًا أكثر من المولّد نفسه.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(seed: int, *names: str) -> int:
    """بذرة ثابتة مشتقة من البذرة الرئيسية واسم الغرض، مثل ``derive_seed(42, "init")``.

    نستعمل blake2b لا ``hash()``، لأن ``hash()`` يتغير بين تشغيل وآخر.
    """
    material = repr((int(seed), *names)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "little")


def make_rng(seed: int, *names: str) -> np.random.Generator:
    """مولّد NumPy (خوارزمية PCG64) لغرض محدد، أو للبذرة نفسها إن لم يُذكر غرض."""
    return np.random.default_rng(derive_seed(seed, *names) if names else int(seed))
