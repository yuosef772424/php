"""بصمات ثابتة للنصوص والملفات.

دالة ``hash()`` المدمجة في بايثون تعطي قيمة مختلفة للنص نفسه في كل تشغيل
(عشوائية عمدًا لأسباب أمنية)، فلا تصلح لإزالة التكرار ولا لتقسيم البيانات تقسيمًا
ثابتًا. نستعمل blake2b من hashlib: القيمة نفسها اليوم وغدًا وعلى أي جهاز.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_hash(data: str | bytes, bits: int = 64) -> int:
    """بصمة عددية ثابتة بطول ``bits`` بت (مضاعف لـ 8، من 8 إلى 512)."""
    if bits % 8 or not 8 <= bits <= 512:
        raise ValueError("bits must be a multiple of 8 between 8 and 512")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=bits // 8).digest(), "little")


def unit_interval(data: str | bytes) -> float:
    """رقم ثابت في المجال [0, 1) مشتق من البصمة.

    نستعمله لتقسيم الوثائق إلى تدريب وتحقق: الوثيقة التي رقمها أقل من 0.02 تذهب
    إلى التحقق، والتقسيم لا يتغير بين تشغيل وآخر.
    """
    return stable_hash(data, 64) / 2**64


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """بصمة SHA-256 لملف كامل، لفحص سلامة نقاط الحفظ والملفات الثنائية."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
