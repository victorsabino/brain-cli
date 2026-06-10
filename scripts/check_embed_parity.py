#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentence-transformers>=3.2",
#   "optimum[onnxruntime]>=1.20",
# ]
# ///
"""Parity gate for BRAIN_EMBED_BACKEND=onnx.

Encodes synthetic sentences with both the torch and ONNX backends of
paraphrase-multilingual-MiniLM-L12-v2 and reports the minimum cosine
similarity between the two vector sets, plus constructor load times.

Why: brain.db vectors are written by whichever backend was active at save
time. If the backends ever diverge (min cosine < 0.999), mixing them puts
queries and stored chunks in different vector spaces and silently degrades
KNN — the gate's job is to make that loud BEFORE it happens.

Run: uv run scripts/check_embed_parity.py
Synthetic data only — nothing from a real brain.db belongs in this file.
"""

from __future__ import annotations
import sys
import time

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
PARITY_FLOOR = 0.999

# ~20 synthetic sentences: EN + PT, technical + casual, short + long —
# shaped like real queries/memories without containing any.
SENTENCES = [
    "fix null pointer exception in payment service retry loop",
    "decided to use reciprocal rank fusion instead of weighted score sums",
    "the deploy failed because the migration ran before the schema backup",
    "como configurar o cache de embeddings para consultas repetidas",
    "lambda cold start latency dropped after switching to provisioned concurrency",
    "remember to rotate the staging database credentials next sprint",
    "o bug de timezone aparece apenas em relatorios gerados depois da meia-noite",
    "chunked vectors keep long documents searchable past the token window",
    "use exponential backoff with jitter for all external API retries",
    "a reuniao de planejamento foi movida para quinta-feira de manha",
    "FTS5 porter stemming conflates indexing and indexed into one term",
    "the queue worker leaked file descriptors under sustained load",
    "preferir tabelas normalizadas a colunas CSV para tags",
    "grep the whole codebase for the pattern before declaring the fix done",
    "cosine distance over normalized vectors equals one minus the dot product",
    "o cliente reportou que o formulario de contato nao envia anexos grandes",
    "cache eviction drops the hundred oldest rows once the cap is exceeded",
    "single file scripts with inline dependencies simplify agent tooling",
    "always snapshot production state before applying a workflow revision",
    "a busca semantica falha silenciosamente quando o modelo nao esta instalado",
]


def load(backend: str | None):
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    kwargs = {"backend": backend} if backend else {}
    model = SentenceTransformer(MODEL, **kwargs)
    return model, time.perf_counter() - t0


def main() -> int:
    torch_model, t_torch = load(None)
    print(f"torch backend load: {t_torch:.2f}s")
    onnx_model, t_onnx = load("onnx")
    print(f"onnx  backend load: {t_onnx:.2f}s")

    a = torch_model.encode(SENTENCES, normalize_embeddings=True)
    b = onnx_model.encode(SENTENCES, normalize_embeddings=True)
    # Normalized vectors → cosine = dot product, row-wise.
    sims = (a * b).sum(axis=1)
    print(f"cosine similarity over {len(SENTENCES)} sentences: "
          f"min={sims.min():.6f} mean={sims.mean():.6f}")

    if sims.min() < PARITY_FLOOR:
        print(
            "\n" + "!" * 72 +
            f"\n!! PARITY FAILURE: min cosine {sims.min():.6f} < {PARITY_FLOOR}."
            "\n!! The ONNX backend does NOT reproduce torch vectors."
            "\n!! Using BRAIN_EMBED_BACKEND=onnx against a torch-built index"
            "\n!! mixes vector spaces and corrupts semantic search results."
            "\n!! You MUST run `brain reindex --full` under the ONNX backend"
            "\n!! (and stay on it) before searching with it.\n" + "!" * 72,
            file=sys.stderr,
        )
        return 1
    print(f"PARITY OK (min >= {PARITY_FLOOR}): backends are interchangeable; "
          "no reindex needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
