"""torch / transformers must stay OUT of the import path of the Ollama serving tier.

`requirements.txt` does not contain torch or transformers -- they live only in
`requirements-finetune.txt`. `server.py` does `import Model_StartUp as ms`, so an
eager `import torch` at module level makes the documented install
(`pip install -r requirements.txt`, README line 301) fail at startup with
ModuleNotFoundError.

It also drags several GB of PyTorch + CUDA libraries into the API container image,
which is paid on every cold start when the tier scales out.

These tests fail if anyone reintroduces a module-level import.
"""
import ast
import os
import sys
import unittest
from unittest import mock

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

HEAVY = {"torch", "transformers"}


def _module_source():
    with open(os.path.join(BASE, "Model_StartUp.py"), encoding="utf-8") as fh:
        return fh.read()


class TestNoEagerHeavyImport(unittest.TestCase):
    """Static guard: nothing heavy may be imported at module level."""

    def test_no_module_level_heavy_import(self):
        tree = ast.parse(_module_source())
        offenders = []
        for node in tree.body:                      # module level ONLY, not nested
            if isinstance(node, ast.Import):
                offenders += [a.name.split(".")[0] for a in node.names
                              if a.name.split(".")[0] in HEAVY]
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in HEAVY:
                    offenders.append(node.module)
        self.assertEqual(
            offenders, [],
            f"module-level import of {offenders} in Model_StartUp.py -- these are not in "
            f"requirements.txt and break `import server` on a standard install",
        )

    def test_heavy_names_are_bound_locally_wherever_used(self):
        """Every function that uses torch/transformers names must bind them itself."""
        src = _module_source()
        tree = ast.parse(src)
        names = ("torch", "AutoTokenizer", "AutoModelForCausalLM", "BitsAndBytesConfig")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.get_source_segment(src, node) or ""
            used = {n for n in names if f"{n}." in body or f"{n}(" in body}
            if not used:
                continue
            bound = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    bound |= {(a.asname or a.name).split(".")[0] for a in sub.names}
                elif isinstance(sub, ast.ImportFrom):
                    bound |= {(a.asname or a.name) for a in sub.names}
                elif isinstance(sub, ast.Assign):
                    bound |= {t.id for t in sub.targets if isinstance(t, ast.Name)}
            self.assertFalse(
                used - bound,
                f"{node.name}() uses {sorted(used - bound)} without binding it locally",
            )


class TestOllamaPathWithoutTorch(unittest.TestCase):
    """Behavioural: the Ollama serving path must work with torch absent."""

    def setUp(self):
        self.assertIsNone(
            __import__("Model_StartUp")._try_torch(),
            "these tests are only meaningful when torch is NOT installed",
        )

    def test_import_succeeds(self):
        import Model_StartUp as ms
        self.assertTrue(hasattr(ms, "generate_response"))

    def test_constants_server_reads_are_available(self):
        """server.py reads these five directly off the module."""
        import Model_StartUp as ms
        for const in ("MAX_NEW_TOKENS", "TEMPERATURE", "TOP_P",
                      "REPETITION_PENALTY", "DO_SAMPLE"):
            self.assertTrue(hasattr(ms, const), f"server.py needs ms.{const}")

    def test_apply_speed_optimizations_is_a_noop(self):
        """Pure optimisation -- absence of torch must never be fatal."""
        import Model_StartUp as ms
        ms.apply_speed_optimizations()          # must not raise

    def test_gguf_load_reaches_ollama_branch_not_torch(self):
        import Model_StartUp as ms
        fake = mock.MagicMock()
        fake.list.return_value = mock.MagicMock(
            models=[mock.MagicMock(model="gemma4:e4b")])
        with mock.patch.object(ms, "_ollama_lib", fake):
            name, kind = ms.load_model_and_tokenizer("/models/gemma4_Q4_K_M.gguf", "gguf")
        self.assertEqual(kind, "ollama")
        self.assertEqual(name, "gemma4")

    def test_generate_response_ollama_branch_never_touches_torch(self):
        import Model_StartUp as ms
        fake = mock.MagicMock()
        fake.chat.return_value = {"message": {"content": "ok"}}
        with mock.patch.dict(sys.modules, {"ollama": fake}):
            out = ms.generate_response(
                "gemma4:e4b", "ollama", [{"role": "user", "content": "hi"}],
                think_mode=False, show_thinking=False, stream=False)
        self.assertIsInstance(out, str)


class TestHuggingFacePathFailsUsefully(unittest.TestCase):
    """When torch really is needed, say so in words a human can act on."""

    def test_generate_response_hf_raises_actionable_error(self):
        import Model_StartUp as ms
        if ms._try_torch() is not None:
            self.skipTest("torch installed; this path is exercised for real")
        with self.assertRaises(RuntimeError) as ctx:
            ms.generate_response("m", mock.MagicMock(), [{"role": "user", "content": "hi"}])
        self.assertIn("requirements-finetune.txt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestHuggingFacePathStillWorksWhenTorchPresent(unittest.TestCase):
    """The lazy import must still FIND torch when it is installed.

    torch is too large to install in CI, so a stub is injected into sys.modules;
    this proves the lookup path, not torch itself.
    """

    def test_try_torch_picks_up_an_installed_torch(self):
        import Model_StartUp as ms
        stub = mock.MagicMock(name="torch")
        with mock.patch.dict(sys.modules, {"torch": stub}):
            self.assertIs(ms._try_torch(), stub)

    def test_apply_speed_optimizations_uses_torch_when_present(self):
        import Model_StartUp as ms
        stub = mock.MagicMock(name="torch")
        stub.cuda.is_available.return_value = True
        with mock.patch.dict(sys.modules, {"torch": stub}):
            ms.apply_speed_optimizations()
        stub.cuda.set_device.assert_called_once_with(0)
        stub.set_grad_enabled.assert_called_once_with(False)
        self.assertTrue(stub.backends.cuda.matmul.allow_tf32)
