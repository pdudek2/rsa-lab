from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


@dataclass
class FactorizationResult:
    bits: int
    n: int
    p: int
    q: int
    time_sec: float
    iterations: int
    success: bool


SMALL_PRIMES: Sequence[int] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in SMALL_PRIMES:
        return True
    if any(n % p == 0 for p in SMALL_PRIMES):
        return False

    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    rng = random.SystemRandom()
    for _ in range(8):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int) -> int:
    rng = random.SystemRandom()
    while True:
        n = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_probable_prime(n):
            return n


def pollards_rho(n: int) -> Tuple[int, int]:
    if n % 2 == 0:
        return 2, 1
    rng = random.SystemRandom()
    total_iters = 0
    for _ in range(10):
        x = rng.randrange(2, n - 1)
        y = x
        c = rng.randrange(1, n - 1)
        d = 1
        while d == 1:
            x = (pow(x, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            y = (pow(y, 2, n) + c) % n
            d = gcd(abs(x - y), n)
            total_iters += 1
            if d == n:
                break
        if 1 < d < n:
            return d, total_iters
    raise RuntimeError("Pollard Rho failed")


def trial_division(n: int) -> Tuple[int, int]:
    if n % 2 == 0:
        return 2, 1
    i = 3
    iters = 0
    while i * i <= n:
        iters += 1
        if n % i == 0:
            return i, iters
        i += 2
    return n, iters


def factor_semiprime(n: int) -> Tuple[int, int, int, bool]:
    factor: Optional[int] = None
    total_iters = 0
    try:
        cand, iters = pollards_rho(n)
        total_iters += iters
        if 1 < cand < n and n % cand == 0:
            factor = cand
    except RuntimeError:
        pass
    if factor is None:
        cand, td_iters = trial_division(n)
        total_iters += td_iters
        if 1 < cand < n and n % cand == 0:
            factor = cand
    if factor is None:
        return n, 1, total_iters, False
    other = n // factor
    p, q = sorted((factor, other))
    success = (p * q == n) and (p != 1) and (p != n)
    return p, q, total_iters, success


def measure_factorizations(bit_sizes: Sequence[int]) -> List[FactorizationResult]:
    results: List[FactorizationResult] = []
    for bits in bit_sizes:
        p_bits = bits // 2
        q_bits = bits - p_bits
        p = generate_prime(p_bits)
        q = generate_prime(q_bits)
        n = p * q
        start = time.perf_counter()
        fp, fq, iterations, success = factor_semiprime(n)
        elapsed = time.perf_counter() - start
        results.append(FactorizationResult(bits, n, fp, fq, elapsed, iterations, success))
    return results


def print_table(results: Sequence[FactorizationResult]) -> None:
    print("bits  time[s]  iterations  success")
    for r in results:
        print(f"{r.bits:>4}  {r.time_sec:>7.4f}  {r.iterations:>10}  {r.success}")


def fit_models(
    results: Sequence[FactorizationResult],
) -> Optional[
    Tuple[
        dict[str, Tuple[Sequence[float], float, Callable]],
        Optional[Tuple[str, Sequence[float], float, Callable]],
    ]
]:
    if curve_fit is None or np is None:
        print("scipy/numpy not available – pomijam dopasowanie modeli.")
        return None

    valid = [r for r in results if r.success and r.time_sec > 0]
    if len(valid) < 2:
        print("Za malo poprawnych danych do dopasowania modeli.")
        return ({}, None)

    x = np.array([r.bits for r in valid], dtype=float)
    y = np.array([r.time_sec for r in valid], dtype=float)

    def linear_func(x, a, b):
        return a * x + b

    def exp_func(x, A, B):
        return A * np.exp(B * x)

    def r_squared(y_true: "np.ndarray", y_pred: "np.ndarray") -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot else 1.0

    models: dict[str, Tuple[Sequence[float], float, Callable]] = {}

    # Model liniowy
    try:
        popt_lin, _ = curve_fit(linear_func, x, y)
        preds = linear_func(x, *popt_lin)
        r2_lin = r_squared(y, preds)
        if r2_lin >= -1.0:
            models["liniowy"] = (popt_lin, r2_lin, linear_func)
            print(f"Model liniowy: parametry {popt_lin} R^2 = {r2_lin:.4f}")
        else:
            print("Model liniowy odrzucony (R^2 ponizej -1).")
    except Exception as exc:
        print(f"Model liniowy nieudany: {exc}")

    # Model wykladniczy (log-space)
    if len(valid) >= 3:
        try:
            log_y = np.log(y)
            if not np.all(np.isfinite(log_y)):
                raise ValueError("log(y) zawiera nieprawidlowe wartosci")

            def log_linear(x, a, b):
                return a * x + b

            popt_log, _ = curve_fit(log_linear, x, log_y)
            A = float(np.exp(popt_log[1]))
            B = float(popt_log[0])
            preds = exp_func(x, A, B)
            if not np.all(np.isfinite(preds)):
                raise ValueError("prognozy modelu wykladniczego sa nieprawidlowe")
            r2_exp = r_squared(y, preds)
            if r2_exp >= -1.0:
                models["wykladniczy"] = ((A, B), r2_exp, exp_func)
                print(f"Model wykladniczy: parametry {[A, B]} R^2 = {r2_exp:.4f}")
            else:
                print("Model wykladniczy odrzucony (R^2 ponizej -1).")
        except Exception as exc:
            print(f"Model wykladniczy nieudany: {exc}")
    else:
        print("Za malo danych dla modelu wykladniczego – pomijam.")

    if not models:
        print("Brak poprawnych dopasowan.")
        return (models, None)

    best_name = max(models, key=lambda k: models[k][1])
    best_params, best_r2, best_func = models[best_name]
    print(f"Lepszy model: {best_name} (R^2 = {best_r2:.4f})")
    best_model = (best_name, best_params, best_r2, best_func)
    return models, best_model


def estimate_times(
    model_data: Tuple[str, Sequence[float], float, Callable],
    targets: Sequence[int],
) -> None:
    name, params, _, func = model_data
    print(f"Prognoza ({name}):")
    for bits in targets:
        estimate = float(func(bits, *params))
        print(f"  {bits:>3} bit: {estimate:.4f} s")


def save_plot(
    results: Sequence[FactorizationResult],
    models: Optional[dict[str, Tuple[Sequence[float], float, Callable]]] = None,
    path: str = "rsa_factorization.png",
) -> None:
    if plt is None:
        print("matplotlib not available – pomijam wykres.")
        return

    bits = [r.bits for r in results]
    times = [r.time_sec for r in results]

    plt.figure(figsize=(6, 4))
    plt.scatter(bits, times, color="black", label="Dane pomiarowe (Twoje wyniki)")

    if models and np is not None:
        xs = np.linspace(min(bits), max(bits), 200)
        if "wykladniczy" in models:
            params, r2, func = models["wykladniczy"]
            ys = func(xs, *params)
            plt.plot(
                xs,
                ys,
                "r-",
                label=f"Dopasowanie wykladnicze (R² = {r2:.2f})",
            )
        if "liniowy" in models:
            params, r2, func = models["liniowy"]
            ys = func(xs, *params)
            plt.plot(
                xs,
                ys,
                "b--",
                label=f"Dopasowanie liniowe (R² = {r2:.2f})",
            )

    ymax = max(times) * 1.2 if times else 1.0
    if ymax <= 0:
        ymax = 1.0
    plt.ylim(0, ymax)
    plt.xlabel("Rozmiar klucza (bity)")
    plt.ylabel("Czas faktoryzacji (s)")
    plt.title("Porownanie modeli wzrostu czasu faktoryzacji RSA")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zapisano wykres {path}")


def main() -> None:
    bit_sizes = [32, 40, 48, 56, 64, 72, 80, 88]
    results = measure_factorizations(bit_sizes)
    print_table(results)

    models = None
    best_model = None

    fitted = fit_models(results)
    if fitted is not None:
        models, best_model = fitted
    else:
        print("Modele niedostepne – brak biblioteki scipy/numpy.")

    if best_model is not None:
        estimate_times(best_model, [96, 128, 256])
    else:
        print("Brak poprawnego modelu – pomijam prognozy.")

    save_plot(results, models)


if __name__ == "__main__":
    main()
