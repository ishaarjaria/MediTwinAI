"""Demo authentication: hashing, the in-memory store, session helpers and honest wording."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import auth


class TestPasswordHashing(unittest.TestCase):
    def test_round_trip_and_wrong_password(self):
        stored = auth.hash_password("correct horse")
        self.assertTrue(auth.verify_password("correct horse", stored))
        self.assertFalse(auth.verify_password("wrong horse", stored))

    def test_hash_is_salted_and_never_contains_the_password(self):
        a, b = auth.hash_password("same-password"), auth.hash_password("same-password")
        self.assertNotEqual(a, b)  # random salt
        self.assertNotIn("same-password", a)
        self.assertTrue(a.startswith("pbkdf2_sha256$"))

    def test_malformed_hashes_fail_closed(self):
        for bad in ("", "x", "a$b$c$d", "pbkdf2_sha256$notanumber$AA==$AA=="):
            self.assertFalse(auth.verify_password("anything", bad), bad)


class TestAccountStore(unittest.TestCase):
    def setUp(self):
        self.store = auth.AccountStore()

    def test_create_then_authenticate(self):
        user = self.store.create("  Ada   Lovelace ", "Ada@Example.com", "secret1")
        self.assertEqual(user.name, "Ada Lovelace")
        self.assertEqual(user.email, "ada@example.com")
        self.assertFalse(user.demo)
        again = self.store.authenticate("ADA@example.com ", "secret1")
        self.assertEqual(again.email, "ada@example.com")

    def test_store_keeps_no_plain_text_password(self):
        self.store.create("Ada", "ada@example.com", "secret1")
        for record in self.store._accounts.values():
            self.assertNotIn("secret1", " ".join(record))

    def test_validation_messages(self):
        cases = [
            (("", "a@b.co", "secret1"), "name"),
            (("Ada", "not-an-email", "secret1"), "e-mail"),
            (("Ada", "a@b.co", "short"), "password"),
        ]
        for args, word in cases:
            with self.assertRaises(auth.AuthError) as ctx:
                self.store.create(*args)
            self.assertIn(word, str(ctx.exception).lower())
        self.assertEqual(len(self.store), 0)

    def test_duplicate_and_demo_address_are_rejected(self):
        self.store.create("Ada", "ada@example.com", "secret1")
        with self.assertRaises(auth.AuthError):
            self.store.create("Ada Two", "ADA@example.com", "secret2")
        with self.assertRaises(auth.AuthError):
            self.store.create("Impostor", auth.DEMO_USER.email, "secret1")

    def test_wrong_password_and_unknown_user_give_the_same_message(self):
        self.store.create("Ada", "ada@example.com", "secret1")
        with self.assertRaises(auth.AuthError) as wrong:
            self.store.authenticate("ada@example.com", "nope-nope")
        with self.assertRaises(auth.AuthError) as unknown:
            self.store.authenticate("nobody@example.com", "secret1")
        self.assertEqual(str(wrong.exception), str(unknown.exception))  # no account enumeration

    def test_accounts_are_per_store_not_global(self):
        self.store.create("Ada", "ada@example.com", "secret1")
        with self.assertRaises(auth.AuthError):
            auth.AccountStore().authenticate("ada@example.com", "secret1")


class TestSessionHelpers(unittest.TestCase):
    def test_sign_in_and_out(self):
        state: dict = {"view": "twin", "pr_last": 1}
        self.assertFalse(auth.is_authenticated(state))
        auth.sign_in(state, auth.DEMO_USER)
        self.assertTrue(auth.is_authenticated(state))
        self.assertEqual(auth.current_user(state), auth.DEMO_USER)
        auth.sign_out(state)
        self.assertFalse(auth.is_authenticated(state))
        self.assertEqual(state, {})  # the whole session is cleared, not just the user

    def test_only_real_user_objects_count_as_signed_in(self):
        for forged in ("admin", {"name": "x"}, True, 1):
            self.assertFalse(auth.is_authenticated({auth.SESSION_KEY: forged}), repr(forged))

    def test_user_presentation(self):
        self.assertEqual(auth.User("Ada Lovelace", "a@b.co").initials, "AL")
        self.assertEqual(auth.User("", "a@b.co").initials, "U")
        self.assertEqual(auth.DEMO_USER.role, "Demo access")
        self.assertTrue(auth.DEMO_USER.demo)


class TestHonestWording(unittest.TestCase):
    def test_notices_say_this_is_a_demo_and_not_production_security(self):
        self.assertIn("Educational", auth.DEMO_NOTICE)
        self.assertIn("real password", auth.DEMO_NOTICE)
        self.assertIn("not production authentication", auth.DEMO_LIMITS)
        doc = auth.__doc__.lower()
        self.assertIn("not production authentication", doc)
        self.assertIn("memory", doc)

    def test_no_secure_claims_in_the_module_or_pages(self):
        for path in (ROOT / "src/ui/auth.py", ROOT / "src/ui/pages/signin.py", ROOT / "src/ui/pages/landing.py"):
            text = path.read_text(encoding="utf8").lower()
            for banned in ("secure login", "securely stored", "bank-grade", "military", "end-to-end encrypted"):
                self.assertNotIn(banned, text, f"{path.name}: {banned}")

    def test_no_external_auth_dependency(self):
        import ast

        tree = ast.parse((ROOT / "src/ui/auth.py").read_text(encoding="utf8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertTrue(imported <= {"__future__", "base64", "hashlib", "hmac", "re", "secrets", "dataclasses", "typing", "streamlit"}, imported)
        for call in ast.walk(tree):  # no use of Streamlit's OIDC login
            if isinstance(call, ast.Attribute):
                self.assertNotIn(call.attr, ("login", "user"), "st.login/st.user are not used")


if __name__ == "__main__":
    unittest.main()
