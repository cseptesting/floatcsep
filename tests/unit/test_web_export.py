import base64
import json

import numpy as np

from floatcsep.postprocess.web.export import _encode_rates, _summary, _outline, STATIC


def test_encode_rates_roundtrip():
    rates = np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.01, 0.02, 0.03]])
    enc = _encode_rates(rates)
    q = np.frombuffer(base64.b64decode(enc["data"]), dtype="<u2").reshape(3, 3)
    cum = np.cumsum(rates[:, ::-1], axis=1)[:, ::-1]

    assert (q[1] == 0).all()
    assert q[0, 2] == 0 and q[0, 1] > 0
    decoded = enc["vmin"] + (q[q > 0] - 1) / 65534 * (enc["vmax"] - enc["vmin"])
    np.testing.assert_allclose(10**decoded, cum[cum > 0], rtol=1e-3)
    np.testing.assert_allclose(enc["totals"], cum.sum(axis=0), atol=1e-4)


def test_summary_shapes():
    cons = _summary({"name": "S", "observed_statistic": -3.0, "quantile": 0.2, "status": "normal",
                     "test_distribution": list(np.linspace(-5, 0, 100))}, "consistency")
    assert cons["n_sim"] == 100 and len(cons["dist_q"]) == 7

    ntest = _summary({"name": "N", "observed_statistic": 7, "quantile": [0.3, 0.8],
                      "test_distribution": ["poisson", 6.5]}, "consistency")
    assert ntest["dist"] == {"type": "poisson", "params": [6.5]}

    ttest = _summary({"name": "T", "observed_statistic": 0.4, "quantile": [1.2, 2.0, 0.05],
                      "test_distribution": [0.1, 0.7]}, "comparative")
    assert ttest["ci"] == [0.1, 0.7]


def test_outline_is_single_polygon():
    xs, ys = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 0.5, 0.1))
    origins = np.column_stack([xs.ravel(), ys.ravel()])
    geom = _outline(origins, 0.1)
    assert geom is None or geom["type"] == "Polygon"


def test_static_files_present():
    for rel in ["index.html", "js/app.js", "css/app.css", "vendor/leaflet/leaflet.js", "vendor/echarts/echarts.min.js"]:
        assert (STATIC / rel).exists(), rel
