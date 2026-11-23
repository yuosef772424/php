"""فحص البيئة: إصدارات بايثون وNumPy وCuPy، وهل يوجد GPU، وهل يعمل الحساب عليه.

أول ما نشغّله في كل جلسة Colab (القسم 8، فحوص ما قبل الإقلاع):
    python scripts/check_env.py
"""

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sifr.core.backend import get_backend, gpu_available  # noqa: E402


def main() -> int:
    print(f"python   {platform.python_version()}")
    print(f"numpy    {np.__version__}")
    try:
        import cupy

        print(f"cupy     {cupy.__version__}")
    except ImportError:
        print("cupy     not installed")

    if gpu_available():
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else name
        free, total = cupy.cuda.runtime.memGetInfo()
        print(f"gpu      {name}, {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")
    else:
        print("gpu      none, CPU backend only")

    # فحص صغير: ضرب مصفوفتين على الواجهة المختارة، ومقارنة الناتج بقيمة معروفة.
    backend = get_backend("auto")
    x = backend.asarray([[1.0, 2.0], [3.0, 4.0]])
    result = backend.to_numpy(x @ x)
    ok = np.allclose(result, [[7.0, 10.0], [15.0, 22.0]])
    print(f"backend  {backend.name} ({backend.dtype}), matmul check: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
