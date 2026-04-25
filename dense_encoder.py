from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import BertModel
from transformers import XLMRobertaTokenizer

_RUNTIME_CACHE: dict[int, tuple[XLMRobertaTokenizer, BertModel, np.ndarray]] = {}


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-9)
    return summed / counts


def build_dense_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    centroids = np.zeros(
        (len(unique_labels), embeddings.shape[1]),
        dtype=np.float32,
    )
    counts = np.zeros(len(unique_labels), dtype=np.int32)
    np.add.at(centroids, inverse_indices, embeddings)
    np.add.at(counts, inverse_indices, 1)
    centroids /= np.maximum(counts[:, None], 1)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.clip(norms, 1e-12, None)
    return unique_labels, centroids


def build_keep_token_ids(
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    extra_base_token_count: int,
    original_vocab_size: int,
    batch_size: int = 64,
) -> np.ndarray:
    keep_ids = set(tokenizer.all_special_ids)
    keep_ids.update(range(min(int(extra_base_token_count), int(original_vocab_size))))

    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=False,
            truncation=False,
        )
        for input_ids in encoded["input_ids"]:
            keep_ids.update(int(token_id) for token_id in input_ids)

    filtered = sorted(
        token_id for token_id in keep_ids if 0 <= int(token_id) < int(original_vocab_size)
    )
    return np.asarray(filtered, dtype=np.int64)


def build_pruned_dense_artifact(
    *,
    model_name_or_path: str,
    train_passage_texts: list[str],
    num_hidden_layers: int,
    extra_base_token_count: int,
) -> dict[str, object]:
    model_path = Path(model_name_or_path)
    if model_path.exists():
        tokenizer = XLMRobertaTokenizer(
            vocab_file=str(model_path / "sentencepiece.bpe.model"),
            bos_token="<s>",
            cls_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
            pad_token="<pad>",
            sep_token="</s>",
            unk_token="<unk>",
        )
        model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(
                model_name_or_path,
                local_files_only=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()

    original_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    keep_token_ids = build_keep_token_ids(
        tokenizer,
        train_passage_texts,
        extra_base_token_count=extra_base_token_count,
        original_vocab_size=original_vocab_size,
    )
    if len(keep_token_ids) == 0:
        raise ValueError("Dense vocabulary pruning kept zero tokens")

    keep_token_ids_tensor = torch.tensor(keep_token_ids, dtype=torch.long)
    word_embeddings = model.embeddings.word_embeddings.weight.detach().cpu()
    pruned_embeddings = word_embeddings.index_select(0, keep_token_ids_tensor)
    model.embeddings.word_embeddings = torch.nn.Embedding(
        num_embeddings=pruned_embeddings.shape[0],
        embedding_dim=pruned_embeddings.shape[1],
        padding_idx=0,
    )
    model.embeddings.word_embeddings.weight.data.copy_(pruned_embeddings)

    if int(num_hidden_layers) < len(model.encoder.layer):
        model.encoder.layer = model.encoder.layer[: int(num_hidden_layers)]

    model.config.vocab_size = int(pruned_embeddings.shape[0])
    model.config.num_hidden_layers = int(num_hidden_layers)

    state_dict = {}
    for name, value in model.state_dict().items():
        if name.endswith("position_ids"):
            continue
        state_dict[name] = value.detach().cpu().numpy().astype(np.float16, copy=False)

    token_id_map = np.full(original_vocab_size, -1, dtype=np.int32)
    token_id_map[keep_token_ids] = np.arange(len(keep_token_ids), dtype=np.int32)
    unk_token_id = int(tokenizer.unk_token_id)
    if token_id_map[unk_token_id] < 0:
        raise ValueError("Dense tokenizer unknown token was pruned")
    token_id_map[token_id_map < 0] = int(token_id_map[unk_token_id])

    sentencepiece_path = Path(tokenizer.vocab_file)
    sentencepiece_bytes = sentencepiece_path.read_bytes()

    return {
        "config": model.config.to_dict(),
        "state_dict": state_dict,
        "token_id_map": token_id_map,
        "sentencepiece_model": sentencepiece_bytes,
        "vocab_size": int(pruned_embeddings.shape[0]),
        "original_vocab_size": int(original_vocab_size),
    }


def _load_dense_runtime(artifact: dict[str, object]) -> tuple[XLMRobertaTokenizer, BertModel, np.ndarray]:
    cache_key = id(artifact)
    cached = _RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    sentencepiece_bytes = artifact["sentencepiece_model"]
    digest = hashlib.sha1(sentencepiece_bytes).hexdigest()
    sp_model_path = Path(tempfile.gettempdir()) / f"bdml-dense-{digest}.model"
    if not sp_model_path.exists() or sp_model_path.read_bytes() != sentencepiece_bytes:
        sp_model_path.write_bytes(sentencepiece_bytes)

    tokenizer = XLMRobertaTokenizer(
        vocab_file=str(sp_model_path),
        bos_token="<s>",
        cls_token="<s>",
        eos_token="</s>",
        mask_token="<mask>",
        pad_token="<pad>",
        sep_token="</s>",
        unk_token="<unk>",
    )

    config = BertConfig(**artifact["config"])
    model = BertModel(config)
    state_dict = {
        name: torch.from_numpy(value.astype(np.float32, copy=False))
        for name, value in artifact["state_dict"].items()
    }
    load_result = model.load_state_dict(state_dict, strict=False)
    unexpected = sorted(load_result.unexpected_keys)
    missing = sorted(load_result.missing_keys)
    allowed_missing = {"embeddings.position_ids"}
    if unexpected or any(key not in allowed_missing for key in missing):
        raise RuntimeError(
            "Failed to load compact dense encoder: "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.eval()
    token_id_map = artifact["token_id_map"]
    cached = (tokenizer, model, token_id_map)
    _RUNTIME_CACHE[cache_key] = cached
    return cached


def encode_texts_with_artifact(
    artifact: dict[str, object],
    texts: list[str],
    *,
    prefix: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    tokenizer, model, token_id_map = _load_dense_runtime(artifact)
    encoded_batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = [prefix + text for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            input_ids = encoded["input_ids"]
            remapped_ids = token_id_map[input_ids]
            attention_mask = torch.from_numpy(encoded["attention_mask"]).long()
            model_inputs = {
                "input_ids": torch.from_numpy(remapped_ids).long(),
                "attention_mask": attention_mask,
            }
            token_type_ids = encoded.get("token_type_ids")
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = torch.from_numpy(token_type_ids).long()
            outputs = model(**model_inputs)
            pooled = mean_pool(outputs.last_hidden_state, attention_mask)
            pooled = F.normalize(pooled, p=2, dim=1)
            encoded_batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))

    if not encoded_batches:
        hidden_size = int(model.config.hidden_size)
        return np.empty((0, hidden_size), dtype=np.float32)
    return np.vstack(encoded_batches)
