"""حفظ الإعدادات وقراءتها بصيغة JSON، مع رفض المفاتيح المجهولة.

لماذا نرفض المفتاح المجهول؟ لو كتبنا في ملف الإعداد ``"learning_rat": 0.001``
بخطأ إملائي وتجاهله البرنامج بصمت، لتدرّب النموذج بالقيمة الافتراضية ونحن نظن
غير ذلك. هذا فقد صامت لمعلومة، والخطأ الصريح يمنعه.

الإعدادات أصناف ``dataclass`` مسطّحة (بلا إعدادات متداخلة)، لتبقى قراءتها بسيطة.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def config_to_dict(config: Any) -> dict[str, Any]:
    """تحويل إعداد (كائن dataclass) إلى قاموس."""
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError("config must be a dataclass instance")
    return dataclasses.asdict(config)


def config_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """بناء إعداد من قاموس، مع خطأ صريح عند أي مفتاح لا يعرفه الصنف."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError("cls must be a dataclass type")
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {unknown}")
    return cls(**data)


def save_config(config: Any, path: str | Path) -> None:
    """حفظ الإعداد في ملف JSON مقروء. العربية تُحفظ كما هي، لا كرموز \\uXXXX."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config_to_dict(config), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def load_config(cls: type[T], path: str | Path) -> T:
    """قراءة إعداد من ملف JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(cls, data)
