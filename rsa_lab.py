# Implementacja RSA na potrzeby zadania laboratoryjnego.
# Autor: Patryk Dudek

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


# ---------------------------------------------------------------------------
# Podstawowe narzedzia arytmetyczne
# ---------------------------------------------------------------------------

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]


def gcd(a: int, b: int) -> int:
    """Najwiekszy wspolny dzielnik."""

    while b:
        a, b = b, a % b
    return a


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Rozszerzony algorytm Euklidesa: zwraca (g, x, y) takie, że ax + by = g."""

    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y


def modinv(a: int, m: int) -> int:
    """Odwrotnosc modularna: liczba x taka, ze a*x ≡ 1 (mod m)."""

    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("Odwrotnosc modularna nie istnieje (a i m nie sa wzglednie pierwsze)")
    return x % m


def modexp(base: int, exp: int, modulus: int) -> int:
    """Potegowanie modularne (square-and-multiply)."""

    if modulus == 1:
        return 0
    result = 1
    base = base % modulus
    while exp > 0:
        if exp & 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        exp >>= 1
    return result


# ---------------------------------------------------------------------------
# Test pierwszosci oraz generowanie liczb pierwszych
# ---------------------------------------------------------------------------


def is_probable_prime(n: int, k: int = 20) -> bool:
    """Probabilistyczny test Millera-Rabina z obsluga malych liczb."""

    if n < 2:
        return False
    if n in SMALL_PRIMES:
        return True
    for p in SMALL_PRIMES:
        if n % p == 0:
            return False

    # zapis n-1 jako d * 2^r
    r = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        r += 1

    rng = random.SystemRandom()
    for _ in range(k):
        a = rng.randrange(2, n - 2)
        x = modexp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = modexp(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int) -> int:
    """Losowanie liczby pierwszej o zadanej liczbie bitow."""

    rng = random.SystemRandom()
    while True:
        # ustawiamy najwyzszy bit i wymuszamy nieparzystosc
        candidate = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_probable_prime(candidate):
            return candidate


# ---------------------------------------------------------------------------
# Struktura na klucz RSA
# ---------------------------------------------------------------------------

@dataclass
class RSAKey:
    p: int
    q: int
    n: int
    phi: int
    e: int
    d: int

    @property
    def bit_length(self) -> int:
        return self.n.bit_length()


# ---------------------------------------------------------------------------
# Generowanie klucza RSA
# ---------------------------------------------------------------------------


class RSAKeyGenerator:
    """Klasa generujaca pary kluczy RSA o zadanym rozmiarze."""

    def __init__(self, bits: int = 768, e_choice: int = 65537) -> None:
        if bits < 128:
            raise ValueError("Potrzebujemy przynajmniej 128 bitow")
        self.bits = bits
        self.e_choice = e_choice
        self.rng = random.SystemRandom()

    def generate(self) -> RSAKey:
        p, q = self._choose_primes()
        n = p * q
        phi = (p - 1) * (q - 1)
        e = self._select_exponent(phi)
        d = modinv(e, phi)
        return RSAKey(p=p, q=q, n=n, phi=phi, e=e, d=d)

    def _choose_primes(self) -> tuple[int, int]:
        half = self.bits // 2
        other = self.bits - half
        p = generate_prime(half)
        q = generate_prime(other)
        while q == p:
            q = generate_prime(other)
        return p, q

    def _select_exponent(self, phi: int) -> int:
        """Dobor e tak, aby gcd(e, phi) == 1."""

        candidates = [self.e_choice, 3, 5, 17, 257, 65537]
        for candidate in candidates:
            if 1 < candidate < phi and gcd(candidate, phi) == 1:
                return candidate
        while True:
            candidate = self.rng.randrange(3, phi - 1)
            if gcd(candidate, phi) == 1:
                return candidate


# ---------------------------------------------------------------------------
# Kodowanie blokow tekstu na liczby i z powrotem
# ---------------------------------------------------------------------------


class BlockCodec:
    """Koduje bloki po 10 znakow jako liczby i odwrotnie."""

    def __init__(self, block_len: int = 10) -> None:
        self.block_len = block_len

    def split_text(self, text: str) -> List[str]:
        """Dzieli tekst na bloki block_len; ostatni dopelnia spacjami."""

        blocks: List[str] = []
        for i in range(0, len(text), self.block_len):
            block = text[i : i + self.block_len]
            if len(block) < self.block_len:
                block = block.ljust(self.block_len)
            blocks.append(block)
        if not blocks:
            blocks.append(" " * self.block_len)
        return blocks

    def block_to_int(self, block: str) -> int:
        """Blok (10 znakow) -> liczba."""

        data = block.encode("utf-8")
        if len(data) != self.block_len:
            raise ValueError("Blok musi miec dokladnie block_len bajtow (ASCII)")
        return int.from_bytes(data, byteorder="big")

    def int_to_block(self, value: int) -> str:
        """Liczba -> blok znakow o dlugosci block_len."""

        data = value.to_bytes(self.block_len, byteorder="big")
        return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Silnik RSA operujacy na blokach
# ---------------------------------------------------------------------------


class RSAEngine:
    """Szyfrowanie i deszyfrowanie blokow tekstu."""

    def __init__(self, key: RSAKey, codec: BlockCodec) -> None:
        self.key = key
        self.codec = codec

    def encrypt_block(self, m: int) -> int:
        """Sprawdza 0 <= m < n, potem wylicza c = m^e mod n."""

        if not 0 <= m < self.key.n:
            raise ValueError("Warunek m < n musi byc spelniony")
        return modexp(m, self.key.e, self.key.n)

    def decrypt_block(self, c: int) -> int:
        """Zwraca m = c^d mod n."""

        return modexp(c, self.key.d, self.key.n)

    def encrypt_text(self, text: str) -> List[int]:
        """Tekst -> bloki -> szyfrogramy."""

        cipher_blocks: List[int] = []
        for block in self.codec.split_text(text):
            m = self.codec.block_to_int(block)
            cipher_blocks.append(self.encrypt_block(m))
        return cipher_blocks

    def decrypt_blocks(self, cipher_blocks: Sequence[int]) -> str:
        """Lista szyfrogramow -> tekst."""

        blocks: List[str] = []
        for c in cipher_blocks:
            m = self.decrypt_block(c)
            blocks.append(self.codec.int_to_block(m))
        return "".join(blocks)


# ---------------------------------------------------------------------------
# Obsluga plikow
# ---------------------------------------------------------------------------


class RSAFileManager:
    """Czytanie i zapisywanie plikow plain.txt / cipher.txt / decrypted.txt."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(".")
        self.plain_path = self.base_dir / "plain.txt"
        self.cipher_path = self.base_dir / "cipher.txt"
        self.decrypted_path = self.base_dir / "decrypted.txt"

    def ensure_plaintext(self) -> None:
        """Tworzy prosty plik plain.txt, jesli jeszcze nie istnieje."""

        if not self.plain_path.exists():
            self.plain_path.write_text("This file demon\nRSA laboratory exercise.\n", encoding="utf-8")

    def read_plaintext(self) -> str:
        return self.plain_path.read_text(encoding="utf-8")

    def write_ciphertext(self, cipher_blocks: Sequence[int]) -> None:
        with self.cipher_path.open("w", encoding="utf-8") as f:
            for c in cipher_blocks:
                f.write(f"{c}\n")

    def read_ciphertext(self) -> List[int]:
        if not self.cipher_path.exists():
            return []
        blocks: List[int] = []
        with self.cipher_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    blocks.append(int(line))
        return blocks

    def write_decrypted(self, text: str) -> None:
        self.decrypted_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Eksperymenty do sprawozdania
# ---------------------------------------------------------------------------


def experiment_m_ge_n(engine: RSAEngine) -> None:
    """Krotki test z przypadkiem m >= n."""

    n = engine.key.n
    m = n + 1
    c = modexp(m, engine.key.e, n)
    m_back = modexp(c, engine.key.d, n)

    print("\nEksperyment: m >= n")
    print("m =", m)
    print("c =", c)
    print("m' =", m_back)


def experiment_identical_blocks(engine: RSAEngine) -> None:
    """Sprawdzenie, czy powtarzajace sie bloki daja ten sam szyfrogram."""

    repeating_block = "HELLO_RSA!"  # dokladnie 10 znakow
    plaintext = repeating_block * 3
    cipher_blocks = engine.encrypt_text(plaintext)

    print("\nEksperyment: identyczne bloki")
    for idx, block in enumerate(cipher_blocks, 1):
        print(f"blok {idx}: {block}")


# ---------------------------------------------------------------------------
# Uwaga o OAEP / paddingu (krotka notatka)
# ---------------------------------------------------------------------------

# W praktyce trzeba dolaczac losowy padding (np. OAEP), tak aby dwa identyczne
# bloki jawne nie dawaly tych samych szyfrogramow. Tutaj tego nie robimy,
# bo na laboratorium wystarcza prosty, podrecznikowy wariant RSA.


# ---------------------------------------------------------------------------
# Funkcja main
# ---------------------------------------------------------------------------


def main() -> None:
    codec = BlockCodec(block_len=10)
    key_generator = RSAKeyGenerator(bits=768, e_choice=65537)
    key = key_generator.generate()
    engine = RSAEngine(key, codec)
    file_manager = RSAFileManager()

    file_manager.ensure_plaintext()
    plaintext = file_manager.read_plaintext()

    print("Klucz RSA:")
    print(f"  |n| = {key.bit_length} bitow")

    cipher_blocks = engine.encrypt_text(plaintext)
    file_manager.write_ciphertext(cipher_blocks)

    cipher_from_file = file_manager.read_ciphertext()
    decrypted = engine.decrypt_blocks(cipher_from_file)
    file_manager.write_decrypted(decrypted)

    original_clean = plaintext.rstrip()
    decrypted_clean = decrypted.rstrip()
    print("Tekst oryginalny i odszyfrowany sa takie same:", original_clean == decrypted_clean)

    experiment_m_ge_n(engine)
    experiment_identical_blocks(engine)


if __name__ == "__main__":
    main()
