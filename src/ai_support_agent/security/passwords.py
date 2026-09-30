"""Password hashing helpers; plaintext passwords must never reach persistence."""

from pwdlib import PasswordHash


_PASSWORD_HASHER = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Create a salted Argon2 password hash."""

    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Safely verify a supplied password against its stored hash."""

    return _PASSWORD_HASHER.verify(password, password_hash)
