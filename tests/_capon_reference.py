"""Independent NumPy reference used by tests and archived real-frame audits."""

import numpy as np

M = np.array([0, -1, -2, -3, -2, -3, -4, -5, -4, -5, -6, -7])
N = np.array([-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1])
ROW = [0, 1, 2, 3, 8, 9, 10, 11]
NU = -np.sin(np.deg2rad(70)) * (1 - np.arange(187) * 0.75 / 70)
MU = -np.sin(np.deg2rad(20)) * (1 - np.arange(27) * 1.5 / 20)


def steering(nu, mu):
    return np.exp(-1j * np.pi * (0.9813 * M * nu + N * mu))


def range_weights(ranges):
    """Reference sets as coefficient vectors, independent of power recurrence."""
    identity = np.eye(ranges)
    left = np.zeros(ranges)
    rows = []
    for stop in (10, 12, 14, 16):
        left = identity[8:stop].sum(axis=0)
        right = left - identity[8] + identity[16]
        rows.append((4, left.copy(), right.copy()))
    for r in range(5, ranges - 4):
        remove_left = r + 3 if r <= 16 else r - 13
        add_left = r + 11 if r <= 8 else r - 5
        remove_right = r + 4 if r < ranges - 8 else r - 12
        add_right = r + 12 if r < ranges - 16 else r - 4
        left = left + identity[add_left] - identity[remove_left]
        right = right + identity[add_right] - identity[remove_right]
        rows.append((r, left.copy(), right.copy()))
    return rows


def cfar(power):
    ranges = power.shape[1]
    weights = range_weights(ranges)
    maxima = power.max(axis=0)
    output = []
    for a in range(2, 185):
        for r, left, right in weights:
            noise = np.float32(min(left @ power[a], right @ power[a])) / np.float32(8)
            cut = power[a, r]
            if cut <= noise * np.float32(5):
                continue
            if a < 20:
                q = 20 - a
                li = list(range(187 - q, 181)) + list(range(0, 12 - q))
                ri = list(range(a + 9, a + 21))
            elif a >= 167:
                q = max(0, 20 - (186 - a - 8))
                li = list(range(a - 12, a))
                ri = list(range(q)) + list(range(a + 9, 187))
            else:
                li, ri = list(range(a - 20, a - 8)), list(range(a + 9, a + 21))
            # Explicit scalar float32 accumulation matches the source's precision.
            ls = sum((power[i, r] for i in li), start=np.float32(0))
            rs = sum((power[i, r] for i in ri), start=np.float32(0))
            normal = cut > min(ls, rs) * np.float32(8 / 12)
            neighbor = cut > max(power[a - 1, r], power[a + 1, r], np.float32(0.4) * maxima[r])
            if normal or neighbor:
                output.append((r, a, float(noise)))
    return output[:150], len(output)


def evaluate(data, doppler_bins):
    """Float64 covariance/solve and independent stage definitions."""
    x = np.asarray(data, dtype=np.complex128)
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.einsum("tir,tjr->rij", x, x.conj()) / len(x)
    a = np.column_stack([steering(u, 0)[ROW] for u in NU])
    power = np.zeros((187, x.shape[2]), dtype=np.float32)
    for r, c in enumerate(cov):
        sub = c[np.ix_(ROW, ROW)]
        scale = sub.trace().real / 8
        if scale:
            loaded = sub + 0.001 * scale * np.eye(8)
            power[:, r] = 1 / np.einsum("ij,ij->j", a.conj(), np.linalg.solve(loaded, a)).real
    candidates, total = cfar(power)
    detections = []
    for r, ai, noise in candidates:
        if noise <= 0:
            continue
        loaded = cov[r] + 0.03 * cov[r].trace().real / 12 * np.eye(12)
        a = np.column_stack([steering(NU[ai], mu) for mu in MU])
        weights = np.linalg.solve(loaded, a)
        spectrum = 1 / np.einsum("ij,ij->j", a.conj(), weights).real
        k = int(np.argmax(spectrum))
        indices = np.array([max(0, k - 1), k, min(26, k + 1)])
        ki = float(indices @ spectrum[indices] / spectrum[indices].sum())
        up = MU[0] + (MU[1] - MU[0]) * ki
        forward2 = 1 - NU[ai] ** 2 - up**2
        if forward2 <= 0:
            continue
        slow = x[:, :, r] @ weights[:, k].conj()
        d = int(np.argmax(abs(np.fft.fft(slow, n=doppler_bins)) ** 2))
        if d > doppler_bins // 2:
            d -= doppler_bins
        detections.append(
            {
                "range_bin": r,
                "azimuth_bin": ai,
                "elevation_bin": k,
                "elevation_interpolated_bin": ki,
                "doppler_bin": d,
                "direction": [float(np.sqrt(forward2)), float(-NU[ai]), float(up)],
                "snr": float(power[ai, r]) / noise,
                "range_noise": noise,
                "elevation_power": float(spectrum[k]),
            }
        )
    return power, detections, total
