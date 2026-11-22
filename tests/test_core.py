"""اختبارات القسم 1: كل أداة مشتركة تفعل ما تقول، ولا تضيّع معلومة بصمت."""

import dataclasses
import hashlib

import numpy as np
import pytest

from sifr.core import backend as be
from sifr.core.config import config_from_dict, load_config, save_config
from sifr.core.hashing import file_digest, stable_hash, unit_interval
from sifr.core.runlog import CSVLogger, plot_curves, read_log
from sifr.core.seeding import derive_seed, make_rng


# --- الواجهة الخلفية -------------------------------------------------------


def test_cpu_backend_is_numpy_float64():
    b = be.get_backend("cpu")
    assert b.name == "numpy" and b.xp is np and b.dtype == np.float64


def test_auto_backend_falls_back_to_cpu_without_gpu(monkeypatch):
    monkeypatch.setattr(be, "gpu_available", lambda: False)
    assert be.get_backend("auto").name == "numpy"


def test_gpu_backend_without_gpu_is_an_explicit_error(monkeypatch):
    # لا تحويل صامت إلى المعالج: من طلب GPU يجب أن يعرف أنه غير موجود.
    monkeypatch.setattr(be, "gpu_available", lambda: False)
    with pytest.raises(RuntimeError):
        be.get_backend("gpu")


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValueError):
        be.get_backend("tpu")


def test_scatter_add_keeps_duplicate_indices():
    # التوكن 1 يظهر مرتين في الدفعة، فيجب أن يُجمع تدرجه مرتين.
    b = be.get_backend("cpu")
    target = np.zeros((3, 2))
    b.scatter_add(target, np.array([1, 1, 2]), np.ones((3, 2)))
    assert target.tolist() == [[0, 0], [2, 2], [1, 1]]


def test_naive_index_add_loses_duplicates():
    # المثال الذي نتجنبه: الكتابة المباشرة تحتفظ بآخر قيمة للموضع المكرر.
    target = np.zeros((3, 2))
    target[np.array([1, 1, 2])] += np.ones((3, 2))
    assert target[1].tolist() == [1, 1]  # كان الصحيح [2, 2]


@pytest.mark.skipif(not be.gpu_available(), reason="needs CuPy and a GPU")
def test_gpu_float32_matches_cpu_float64():
    cpu, gpu = be.get_backend("cpu"), be.get_backend("gpu")
    a = make_rng(0, "matmul").standard_normal((64, 32))
    expected = a @ a.T
    got = gpu.to_numpy(gpu.asarray(a) @ gpu.asarray(a).T)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)
    assert cpu.dtype == np.float64 and gpu.dtype == np.float32


# --- البذور ---------------------------------------------------------------


def test_same_seed_same_numbers():
    first = make_rng(42, "init").standard_normal(5)
    second = make_rng(42, "init").standard_normal(5)
    assert np.array_equal(first, second)


def test_different_purposes_get_different_streams():
    init = make_rng(42, "init").standard_normal(5)
    data = make_rng(42, "data").standard_normal(5)
    assert not np.array_equal(init, data)


def test_derive_seed_is_stable_across_runs():
    # قيمة محسوبة مسبقًا. إن تغيرت، تغيرت معها كل التجارب السابقة.
    assert derive_seed(42, "init") == 16928586722087840697


# --- البصمات --------------------------------------------------------------


def test_stable_hash_known_value():
    # blake2b معيار ثابت، فالبصمة نفسها على أي جهاز وفي أي تشغيل.
    assert stable_hash("مرحبا") == 17551019120239679226


def test_stable_hash_rejects_bad_size():
    with pytest.raises(ValueError):
        stable_hash("x", bits=12)


def test_unit_interval_is_in_range_and_balanced():
    values = [unit_interval(f"doc-{i}") for i in range(2000)]
    assert all(0.0 <= v < 1.0 for v in values)
    below_half = sum(v < 0.5 for v in values) / len(values)
    assert 0.45 < below_half < 0.55


def test_file_digest_matches_sha256(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"sifr")
    assert file_digest(path) == hashlib.sha256(b"sifr").hexdigest()


# --- الإعدادات ------------------------------------------------------------


@dataclasses.dataclass
class _Demo:
    learning_rate: float = 1e-3
    name: str = "صفر"


def test_config_round_trip_keeps_arabic(tmp_path):
    path = tmp_path / "cfg.json"
    save_config(_Demo(learning_rate=6e-4), path)
    assert load_config(_Demo, path) == _Demo(learning_rate=6e-4)
    assert "صفر" in path.read_text(encoding="utf-8")


def test_config_rejects_unknown_keys():
    # خطأ إملائي في اسم الإعداد يجب ألا يُتجاهل بصمت.
    with pytest.raises(KeyError):
        config_from_dict(_Demo, {"learning_rat": 0.1})


# --- السجلات --------------------------------------------------------------


def test_csv_logger_appends_after_resume(tmp_path):
    path = tmp_path / "log.csv"
    CSVLogger(path, ["step", "loss"]).log({"step": 1, "loss": 9.01})
    CSVLogger(path, ["step", "loss"]).log({"step": 2, "loss": 8.5})  # استئناف
    assert [row["step"] for row in read_log(path)] == ["1", "2"]


def test_csv_logger_rejects_unknown_column(tmp_path):
    log = CSVLogger(tmp_path / "log.csv", ["step", "loss"])
    with pytest.raises(KeyError):
        log.log({"step": 1, "los": 1.0})


def test_csv_logger_rejects_mismatched_header(tmp_path):
    path = tmp_path / "log.csv"
    CSVLogger(path, ["step", "loss"])
    with pytest.raises(ValueError):
        CSVLogger(path, ["step", "val_loss"])


def test_plot_curves_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "log.csv"
    log = CSVLogger(path, ["step", "loss"])
    for step, loss in [(1, 9.0), (2, 7.5), (3, 6.8)]:
        log.log({"step": step, "loss": loss})
    out = plot_curves(path, tmp_path / "loss.png", x="step", ys=["loss"])
    assert out.exists() and out.stat().st_size > 0
