"""Program edukacyjny pokazujacy od podstaw dzialanie RSA z opisanymi
obowiazkowymi eksperymentami i obsluga plikow.  Wszystkie elementy zostaly
odseparowane w klasy/funkcje o jednej odpowiedzialnosci, aby latwiej bylo
powiazac kod z zasadami SOLID/DRY/KISS."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


# ---------------------------------------------------------------------------
# Podstawowe narzedzia arytmetyczne
# ---------------------------------------------------------------------------


@dataclass
class RSAKey:
    """Prosty kontener na parametry klucza RSA."""

    p: int
    q: int
    n: int
    phi: int
    e: int
    d: int

    @property
    def bit_length(self) -> int:
        """Liczba bitow modulu n, pomocna przy raportowaniu rozmiaru klucza."""

        return self.n.bit_length()


SMALL_PRIMES: Sequence[int] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def gcd(a: int, b: int) -> int:
    """Najwiekszy wspolny dzielnik liczony klasycznym algorytmem Euklidesa."""

    while b:
        a, b = b, a % b
        return abs(a)


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Rozszerzony algorytm Euklidesa zwracajacy (g, x, y) z rownania ax + by = g."""

    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return (g, x, y)


def modinv(a: int, m: int) -> int:
    """Odwrotnosc modularna istnieje tylko gdy gcd(a, m) == 1."""

    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("Odwrotnosc modularna nie istnieje (a i m nie sa wzglednie pierwsze)")
    return x % m


def modexp(base: int, exp: int, modulus: int) -> int:
    """Potegowanie modularne metodą square-and-multiply (bezpiecznie i wydajnie)."""

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
    """Probabilistyczny test Millera-Rabina ze specjalnym traktowaniem malych liczb."""

    if n < 2:
        return False
    if n in SMALL_PRIMES:
        return True
    if any((n % p) == 0 for p in SMALL_PRIMES):
        return False
    if n % 2 == 0:
        return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    rng = random.SystemRandom()
    for _ in range(k):
        a = rng.randrange(2, n - 1)
        x = modexp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = modexp(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int) -> int:
    """Generator liczb pierwszych o zadanej dlugosci bitowej z wymuszeniem parzystosci."""

    if bits < 2:
        raise ValueError("Minimalna dlugosc to 2 bity")
    rng = random.SystemRandom()
    while True:
        candidate = rng.getrandbits(bits)
        candidate |= (1 << (bits - 1))
        candidate |= 1
        if is_probable_prime(candidate):
            return candidate


# ---------------------------------------------------------------------------
# Generowanie klucza RSA (zasady SOLID: osobna klasa za to odpowiada)
# ---------------------------------------------------------------------------


class RSAKeyGenerator:
    """Klasa przygotowujaca pary kluczy RSA o zadanym rozmiarze."""

    def __init__(self, bits: int = 768, e_choice: int = 65537) -> None:
        if bits < 128:
            raise ValueError("Do labu potrzebujemy przynajmniej 128 bitow")
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
        while p == q:
            q = generate_prime(other)
        return p, q

    def _select_exponent(self, phi: int) -> int:
        """e musi byc wzglednie pierwsze z phi, aby istnial klucz prywatny."""

        candidates = []
        if self.e_choice is not None:
            candidates.append(self.e_choice)
        candidates.extend([3, 5, 17, 257, 65537])
        for candidate in candidates:
            if candidate is None or candidate < 3:
                continue
            if gcd(candidate, phi) == 1:
                # W praktyce najczesciej wybiera sie 65537 ze wzgledu na szybkosc oraz bezpieczenstwo.
                return candidate
        while True:
            candidate = self.rng.randrange(3, phi - 1)
            if candidate % 2 == 0:
                candidate += 1
            if gcd(candidate, phi) == 1:
                return candidate


# ---------------------------------------------------------------------------
# Kodowanie blokow tekstowych jako liczb calkowitych
# ---------------------------------------------------------------------------


class BlockCodec:
    """Obsługa konwersji tekst <-> bloki liczbowe o stalym rozmiarze."""

    def __init__(self, block_len: int = 10) -> None:
        self.block_len = block_len

    def split_text(self, text: str) -> List[str]:
        """Zwraca bloki po block_len znakow, ostatni blok wypelnia spacjami."""

        blocks: List[str] = []
        for i in range(0, len(text), self.block_len):
            chunk = text[i : i + self.block_len]
            if len(chunk) < self.block_len:
                chunk = chunk.ljust(self.block_len)
            blocks.append(chunk)
        if not blocks:
            blocks.append(" " * self.block_len)
        return blocks

    def block_to_int(self, block: str) -> int:
        """Kodowanie UTF-8 -> int. Przy stalym rozmiarze odwzorowanie jest bijekcja."""

        data = block.encode("utf-8")
        if len(data) != self.block_len:
            raise ValueError("Blok musi miec dokladnie block_len bajtow")
        # Przy 10 bajtach otrzymujemy maksymalnie 80 bitow, co jest znacznie mniej niz 768 bitow n,
        # wiec typowy blok zawsze spelni warunek m < n.
        return int.from_bytes(data, byteorder="big")

    def int_to_block(self, value: int) -> str:
        """Odwrotna konwersja: liczba -> bajty -> tekst."""

        data = value.to_bytes(self.block_len, byteorder="big")
        return data.decode("utf-8")

    def join_blocks(self, blocks: Sequence[str]) -> str:
        """Laczy bloki w pojedynczy lancuch bez usuwania wypelnien."""

        return "".join(blocks)


# ---------------------------------------------------------------------------
# Silnik RSA odpowiadajacy za szyfrowanie i deszyfrowanie
# ---------------------------------------------------------------------------


class RSAEngine:
    """Klasa wykonujaca operacje RSA na blokach z wykorzystaniem zadanego klucza."""

    def __init__(self, key: RSAKey, codec: BlockCodec) -> None:
        self.key = key
        self.codec = codec

    def encrypt_block(self, m: int) -> int:
        """Pojedynczy blok: sprawdzamy 0 <= m < n, potem potegujemy modularnie."""

        if not 0 <= m < self.key.n:
            raise ValueError("Warunek m < n jest konieczny, inaczej tracimy informacje")
        return modexp(m, self.key.e, self.key.n)

    def decrypt_block(self, c: int) -> int:
        """Deszyfrowanie pojedynczego bloku."""

        return modexp(c, self.key.d, self.key.n)

    def encrypt_text(self, text: str) -> List[int]:
        """Konwersja tekstu do blokow liczbowych, a nastepnie szyfrowanie kazdego z nich."""

        cipher_blocks: List[int] = []
        for block in self.codec.split_text(text):
            m = self.codec.block_to_int(block)
            cipher_blocks.append(self.encrypt_block(m))
        return cipher_blocks

    def decrypt_blocks(self, cipher_blocks: Sequence[int]) -> str:
        """Deszyfruje liste liczb, zamienia na tekst i laczy bloki."""

        blocks: List[str] = []
        for c in cipher_blocks:
            m = self.decrypt_block(c)
            blocks.append(self.codec.int_to_block(m))
        return self.codec.join_blocks(blocks)


# ---------------------------------------------------------------------------
# Obsluga plikow: osobna klasa dla czytania/zapisu (Single Responsibility)
# ---------------------------------------------------------------------------


class RSAFileManager:
    """Odpowiada za pliki plain.txt, cipher.txt i decrypted.txt."""

    def __init__(self, plain: str = "plain.txt", cipher: str = "cipher.txt", decrypted: str = "decrypted.txt") -> None:
        self.plain_path = Path(plain)
        self.cipher_path = Path(cipher)
        self.decrypted_path = Path(decrypted)
        self.sample_text = (
            "Sample plaintext for the RSA lab.\n"
            "Create your own plain.txt to experiment with different messages."
        )

    def ensure_plaintext(self) -> None:
        """Jesli plain.txt nie istnieje tworzymy przykladowy plik, aby aplikacja byla samowystarczalna."""

        if not self.plain_path.exists():
            self.plain_path.write_text(self.sample_text, encoding="utf-8")

    def read_plaintext(self) -> str:
        return self.plain_path.read_text(encoding="utf-8")

    def write_ciphertext(self, cipher_blocks: Sequence[int]) -> None:
        with self.cipher_path.open("w", encoding="utf-8") as fh:
            for block in cipher_blocks:
                fh.write(f"{block}\n")

    def read_ciphertext(self) -> List[int]:
        blocks: List[int] = []
        with self.cipher_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                blocks.append(int(line))
        return blocks

    def write_decrypted(self, text: str) -> None:
        self.decrypted_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Eksperymenty wymagane w instrukcji laboratoryjnej
# ---------------------------------------------------------------------------


def experiment_m_ge_n(engine: RSAEngine) -> None:
    """Pokazuje, ze RSA zawsze dziala modulo n i dlatego wymagamy m < n."""

    n = engine.key.n
    m = n + 1
    c = modexp(m, engine.key.e, n)
    m_back = modexp(c, engine.key.d, n)
    print("[Eksperyment m >= n]")
    print(f"m = n + 1 = {m}")
    print(f"c = m^e mod n = {c}")
    print(f"m_back = c^d mod n = {m_back}")
    print("Odzyskujemy tylko m mod n, wiec nadmiarowe informacje przepadaja.\n")


def experiment_identical_blocks(engine: RSAEngine) -> None:
    """Demonstracja deterministycznosci tekstbookowego RSA."""

    repeating_block = "HELLO_RSA!"  # dokladnie 10 znakow
    plaintext = repeating_block * 3
    cipher_blocks = engine.encrypt_text(plaintext)
    print("[Eksperyment identyczne bloki]")
    print(f"Plik testowy sklada sie z bloku '{repeating_block}' powtorzonego 3 razy")
    print("Szyfrogramy (w postaci liczb calkowitych):")
    for idx, block in enumerate(cipher_blocks, 1):
        print(f"  Blok {idx}: {block}")
    print("Kazdy szyfrogram jest identyczny, bo czyste RSA nie dodaje losowosci.\n")


# ---------------------------------------------------------------------------
# Uwaga o OAEP/paddingu (tylko komentarz)
# ---------------------------------------------------------------------------

# Deterministyczne RSA zdradza wzorce: jesli napastnik widzi takie same bloki
# szyfrogramu, to wie ze odpowiadaja im takie same bloki jawne. Schematy paddingu
# takie jak OAEP dodaja losowosc i strukture do danych zanim zostana podniesione
# do potegi. Ten dodatek sprawia, ze nawet ten sam komunikat logiczny daje inne
# bloki liczbowe, co utrudnia analiza wzorcow i zapewnia poufnosc semantyczna.


# ---------------------------------------------------------------------------
# Funkcja main laczaca wszystkie elementy
# ---------------------------------------------------------------------------


def main() -> None:
    codec = BlockCodec(block_len=10)
    key_generator = RSAKeyGenerator(bits=768, e_choice=65537)
    key = key_generator.generate()
    engine = RSAEngine(key, codec)
    file_manager = RSAFileManager()

    file_manager.ensure_plaintext()
    plaintext = file_manager.read_plaintext()

    print("Wygenerowano klucz RSA:")
    print(f"  dlugosc n (bity): {key.bit_length}")
    print(f"  publiczny wykladnik e: {key.e}")

    cipher_blocks = engine.encrypt_text(plaintext)
    file_manager.write_ciphertext(cipher_blocks)
    print(f"Zaszyfrowano {len(cipher_blocks)} blokow i zapisano do {file_manager.cipher_path}")

    cipher_from_file = file_manager.read_ciphertext()
    decrypted = engine.decrypt_blocks(cipher_from_file)
    file_manager.write_decrypted(decrypted)
    print(f"Odszyfrowane bloki zapisano do {file_manager.decrypted_path}")

    original_clean = plaintext.rstrip()
    decrypted_clean = decrypted.rstrip()
    success = original_clean == decrypted_clean
    print("Czy teksty sa identyczne (po usunieciu koncowych spacji)?", success)
    print("Podglad oryginalu:", original_clean[:40])
    print("Podglad deszyfr.:", decrypted_clean[:40])

    experiment_m_ge_n(engine)
    experiment_identical_blocks(engine)


if __name__ == "__main__":
    main()

