from __future__ import annotations

import gzip
import html
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from dense_encoder import build_dense_centroids
from dense_encoder import build_pruned_dense_artifact
from dense_encoder import encode_texts_with_artifact

TEXT_COLUMNS = [
    "vendor_name",
    "vendor_code",
    "title",
    "description",
    "shop_category_name",
]
OVERRIDE_RULES = [
    ("shop_category_name", "title"),
    ("vendor_code",),
    ("vendor_name", "vendor_code"),
    ("shop_category_name", "vendor_code"),
    ("vendor_name", "title"),
    ("shop_category_name",),
    ("title",),
]
OVERRIDE_MAJORITY_RULES = {
    ("shop_category_name",),
}
OVERRIDE_MAJORITY_MIN_COUNT = 5
OVERRIDE_MAJORITY_MIN_SHARE = 0.98
USE_NORMALIZED_UNIQUE_OVERRIDES = False
_VC_NORM_RE = re.compile(r"[^0-9a-zа-я]+", re.IGNORECASE)
_TITLE_PUNCT_RE = re.compile(r"[^0-9a-zа-я]+", re.IGNORECASE)
_TITLE_NUM_RE = re.compile(r"\d+")
USE_VENDOR_CODE_NORM_DISAMBIGUATION = True
VENDOR_CODE_NORM_MIN_LEN = 5
VENDOR_CODE_NORM_MAX_CANDIDATES = 7
VENDOR_CODE_NORM_MIN_GAIN = 0.0
WORD_VECTORIZER_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "max_features": 100_000,
    "dtype": np.float32,
}
CHAR_VECTORIZER_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "sublinear_tf": True,
    "max_features": 100_000,
    "dtype": np.float32,
}
TITLE_WORD_VECTORIZER_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "max_features": 100_000,
    "dtype": np.float32,
}
TITLE_CHAR_VECTORIZER_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "sublinear_tf": True,
    "max_features": 100_000,
    "dtype": np.float32,
}
USE_TITLE_VIEW = False
BATCH_SIZE = 256
MAIN_CHAR_BLOCK_WEIGHT = 1.0
TITLE_CHAR_BLOCK_WEIGHT = 1.0
VENDOR_REPEATS = 1
CODE_REPEATS = 1
TITLE_REPEATS = 5
SHOP_CATEGORY_REPEATS = 5
DESCRIPTION_REPEATS = 1
TITLE_VIEW_INCLUDE_SHOP_CATEGORY = False
TITLE_VIEW_INCLUDE_VENDOR_NAME = False
TITLE_VIEW_SHOP_CATEGORY_REPEATS = 2
TITLE_VIEW_VENDOR_NAME_REPEATS = 1
TITLE_USE_BM25 = False
BM25_K1 = 1.6
BM25_B = 0.75
UNCERTAIN_MARGIN_THRESHOLD = 0.5
UNCERTAIN_TOP_DEPARTMENTS = 8
TITLE_BONUS_WEIGHT = 0.2
DEPARTMENT_SYNC_SCORE_GAP_THRESHOLD = 0.5
DEPARTMENT_SCORE_BONUS_WEIGHT_FOR_CATEGORY = 0.0
CATEGORY_CENTROID_DEPT_SHRINK_ALPHA = 0.0
_DENSE_LOCAL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots/"
    / "c007d7ef6fd86656326059b28395a7a03a7c5846"
)
DENSE_BACKBONE_PATH = (
    str(_DENSE_LOCAL_SNAPSHOT)
    if _DENSE_LOCAL_SNAPSHOT.exists()
    else "intfloat/multilingual-e5-small"
)
DENSE_NUM_HIDDEN_LAYERS = 3
DENSE_EXTRA_BASE_TOKEN_COUNT = 4096
DENSE_QUERY_PREFIX = "query: "
DENSE_PASSAGE_PREFIX = "passage: "
DENSE_MAX_LENGTH = 96
DENSE_BATCH_SIZE = 64
DENSE_CATEGORY_SCORE_WEIGHT = 0.12
DENSE_DEPARTMENT_SCORE_WEIGHT = 0.2
DENSE_DESCRIPTION_MAX_CHARS = 0
DEPARTMENT_MODEL_PARAMS = {
    "class_weight": "balanced",
    "C": 0.4,
    "random_state": 42,
}


class BM25Vectorizer:
    def __init__(
        self,
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        dtype=np.float32,
        **count_vectorizer_params,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.dtype = dtype
        self._cv = CountVectorizer(**count_vectorizer_params)
        self._idf: np.ndarray | None = None
        self._avgdl: float | None = None

    def fit_transform(self, raw_documents: list[str]) -> sparse.csr_matrix:
        counts = self._cv.fit_transform(raw_documents).tocsr()
        self._init_idf_and_avgdl(counts)
        return self._bm25_transform_from_counts(counts)

    def transform(self, raw_documents: list[str]) -> sparse.csr_matrix:
        if self._idf is None or self._avgdl is None:
            raise RuntimeError("BM25Vectorizer is not fitted")
        counts = self._cv.transform(raw_documents).tocsr()
        return self._bm25_transform_from_counts(counts)

    def _init_idf_and_avgdl(self, counts: sparse.csr_matrix) -> None:
        n_docs = counts.shape[0]
        df = np.diff(counts.tocsc().indptr).astype(np.float64, copy=False)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self._idf = idf.astype(self.dtype, copy=False)
        doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64, copy=False)
        self._avgdl = float(doc_len.mean()) if n_docs else 0.0

    def _bm25_transform_from_counts(self, counts: sparse.csr_matrix) -> sparse.csr_matrix:
        if self._idf is None or self._avgdl is None:
            raise RuntimeError("BM25Vectorizer is not fitted")
        counts = counts.astype(np.float32, copy=True)
        doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32, copy=False)
        avgdl = float(self._avgdl) if self._avgdl else 1.0
        k1 = float(self.k1)
        b = float(self.b)

        # Per-document normalizer: k1 * (1 - b + b * dl/avgdl)
        norm = k1 * (1.0 - b + b * (doc_len / avgdl))
        indptr = counts.indptr
        indices = counts.indices
        data = counts.data
        idf = self._idf

        for row in range(counts.shape[0]):
            start, end = int(indptr[row]), int(indptr[row + 1])
            if start == end:
                continue
            tf = data[start:end]
            cols = indices[start:end]
            denom = tf + norm[row]
            data[start:end] = idf[cols] * (tf * (k1 + 1.0) / denom)
        return counts.tocsr()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def clean_vendor_code(value: object) -> str:
    text = clean_text(value)
    text = _VC_NORM_RE.sub("", text)
    if len(text) < VENDOR_CODE_NORM_MIN_LEN:
        return ""
    return text


def canonicalize_title(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = _TITLE_PUNCT_RE.sub(" ", text)
    text = _TITLE_NUM_RE.sub("#", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(path, sep="\t")
    for column in TEXT_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = ""
    return dataset


def build_search_texts(dataframe: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    rows = dataframe[TEXT_COLUMNS].itertuples(index=False, name=None)
    for vendor_name, vendor_code, title, description, shop_category_name in rows:
        c_vendor = clean_text(vendor_name)
        c_code = clean_text(vendor_code)
        c_title = clean_text(title)
        c_desc = clean_text(description)
        c_shopcat = clean_text(shop_category_name)

        parts = []
        for _ in range(VENDOR_REPEATS):
            parts.append("vendor " + c_vendor)
        for _ in range(CODE_REPEATS):
            parts.append("code " + c_code)
        for _ in range(TITLE_REPEATS):
            parts.append("title " + c_title)
        for _ in range(SHOP_CATEGORY_REPEATS):
            parts.append("shopcat " + c_shopcat)
        for _ in range(DESCRIPTION_REPEATS):
            parts.append("desc " + c_desc)

        texts.append(" ".join(parts))
    return texts


def build_title_texts(dataframe: pd.DataFrame) -> list[str]:
    if not TITLE_VIEW_INCLUDE_SHOP_CATEGORY and not TITLE_VIEW_INCLUDE_VENDOR_NAME:
        return dataframe["title"].map(clean_text).tolist()

    texts: list[str] = []
    rows = dataframe[["title", "shop_category_name", "vendor_name"]].itertuples(
        index=False, name=None
    )
    for title, shop_category_name, vendor_name in rows:
        c_title = clean_text(title)
        c_shopcat = clean_text(shop_category_name)
        c_vendor = clean_text(vendor_name)
        parts = ["title " + c_title]
        if TITLE_VIEW_INCLUDE_SHOP_CATEGORY:
            for _ in range(TITLE_VIEW_SHOP_CATEGORY_REPEATS):
                parts.append("shopcat " + c_shopcat)
        if TITLE_VIEW_INCLUDE_VENDOR_NAME:
            for _ in range(TITLE_VIEW_VENDOR_NAME_REPEATS):
                parts.append("vendor " + c_vendor)
        texts.append(" ".join(parts))
    return texts


def build_dense_texts(dataframe: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    rows = dataframe[TEXT_COLUMNS].itertuples(index=False, name=None)
    for vendor_name, vendor_code, title, description, shop_category_name in rows:
        parts = []
        c_title = clean_text(title)
        c_shopcat = clean_text(shop_category_name)
        c_vendor = clean_text(vendor_name)
        c_code = clean_text(vendor_code)
        c_desc = clean_text(description)

        if c_title:
            parts.append(f"title: {c_title}")
        if c_shopcat:
            parts.append(f"shop category: {c_shopcat}")
        if c_vendor:
            parts.append(f"brand: {c_vendor}")
        if c_code:
            parts.append(f"code: {c_code}")
        if DENSE_DESCRIPTION_MAX_CHARS and c_desc:
            parts.append(f"description: {c_desc[:DENSE_DESCRIPTION_MAX_CHARS]}")

        texts.append(". ".join(parts))
    return texts


def build_rule_keys(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    key_parts = [dataframe[column].map(clean_text) for column in columns]
    key = key_parts[0]
    for part in key_parts[1:]:
        key = key + "||" + part
    return key.to_numpy(dtype=object)


def has_meaningful_key(key: object) -> bool:
    return str(key).strip("|") != ""


def build_unique_category_map(
    dataframe: pd.DataFrame, columns: tuple[str, ...]
) -> dict[str, int]:
    temp = dataframe[list(columns) + ["category_id"]].copy()
    temp["key"] = build_rule_keys(temp, columns)
    temp = temp[temp["key"].map(has_meaningful_key)]
    counts = temp.groupby("key")["category_id"].nunique()
    unique_keys = counts[counts == 1].index
    if len(unique_keys) == 0:
        return {}
    unique_rows = temp[temp["key"].isin(unique_keys)].drop_duplicates("key")
    return dict(zip(unique_rows["key"], unique_rows["category_id"]))


def build_normalized_unique_category_map(
    dataframe: pd.DataFrame,
    column: str,
    normalizer,
) -> dict[str, int]:
    temp = dataframe[[column, "category_id"]].copy()
    temp["key"] = temp[column].map(normalizer)
    temp = temp[temp["key"].map(has_meaningful_key)]
    counts = temp.groupby("key")["category_id"].nunique()
    unique_keys = counts[counts == 1].index
    if len(unique_keys) == 0:
        return {}
    unique_rows = temp[temp["key"].isin(unique_keys)].drop_duplicates("key")
    return dict(zip(unique_rows["key"], unique_rows["category_id"]))


def build_normalized_candidates_map(
    dataframe: pd.DataFrame,
    column: str,
    normalizer,
    *,
    max_candidates: int,
) -> dict[str, np.ndarray]:
    temp = dataframe[[column, "category_id"]].copy()
    temp["key"] = temp[column].map(normalizer)
    temp = temp[temp["key"].map(has_meaningful_key)]
    if len(temp) == 0:
        return {}
    grouped = temp.groupby("key")["category_id"].unique()
    result: dict[str, np.ndarray] = {}
    for key, categories in grouped.items():
        cats = np.asarray(categories, dtype=np.int32)
        if len(cats) <= 1:
            continue
        if len(cats) > int(max_candidates):
            continue
        result[str(key)] = cats
    return result

def build_majority_category_map(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    min_count: int | None = None,
    min_share: float | None = None,
) -> dict[str, int]:
    if min_count is None:
        min_count = int(OVERRIDE_MAJORITY_MIN_COUNT)
    if min_share is None:
        min_share = float(OVERRIDE_MAJORITY_MIN_SHARE)
    temp = dataframe[list(columns) + ["category_id"]].copy()
    temp["key"] = build_rule_keys(temp, columns)
    temp = temp[temp["key"].map(has_meaningful_key)]
    if len(temp) == 0:
        return {}

    grouped = temp.groupby("key")["category_id"]
    total = grouped.count()
    mode = grouped.agg(lambda x: x.value_counts().index[0])
    mode_count = grouped.agg(lambda x: x.value_counts().iloc[0])
    share = mode_count / total
    mask = (total >= int(min_count)) & (share >= float(min_share))
    if not bool(mask.any()):
        return {}
    return dict(zip(mode[mask].index, mode[mask].astype(int).values))

def fit_feature_blocks(
    texts: list[str],
) -> tuple[TfidfVectorizer, TfidfVectorizer, sparse.csr_matrix]:
    word_vectorizer = TfidfVectorizer(**WORD_VECTORIZER_PARAMS)
    char_vectorizer = TfidfVectorizer(**CHAR_VECTORIZER_PARAMS)
    word_features = word_vectorizer.fit_transform(texts)
    char_features = char_vectorizer.fit_transform(texts)
    if MAIN_CHAR_BLOCK_WEIGHT != 1.0:
        char_features = char_features.multiply(np.float32(MAIN_CHAR_BLOCK_WEIGHT))
    features = sparse.hstack(
        [word_features, char_features],
        format="csr",
        dtype=np.float32,
    )
    return word_vectorizer, char_vectorizer, features


def transform_feature_blocks(
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    texts: list[str],
) -> sparse.csr_matrix:
    word_features = word_vectorizer.transform(texts)
    char_features = char_vectorizer.transform(texts)
    if MAIN_CHAR_BLOCK_WEIGHT != 1.0:
        char_features = char_features.multiply(np.float32(MAIN_CHAR_BLOCK_WEIGHT))
    return sparse.hstack(
        [word_features, char_features],
        format="csr",
        dtype=np.float32,
    )


def fit_title_feature_blocks(
    texts: list[str],
) -> tuple[object, object, sparse.csr_matrix]:
    if not USE_TITLE_VIEW:
        empty = sparse.csr_matrix((len(texts), 0), dtype=np.float32)
        return None, None, empty

    if TITLE_USE_BM25:
        word_vectorizer = BM25Vectorizer(
            k1=BM25_K1,
            b=BM25_B,
            analyzer=TITLE_WORD_VECTORIZER_PARAMS["analyzer"],
            ngram_range=TITLE_WORD_VECTORIZER_PARAMS["ngram_range"],
            max_features=TITLE_WORD_VECTORIZER_PARAMS["max_features"],
            dtype=TITLE_WORD_VECTORIZER_PARAMS["dtype"],
        )
        char_vectorizer = BM25Vectorizer(
            k1=BM25_K1,
            b=BM25_B,
            analyzer=TITLE_CHAR_VECTORIZER_PARAMS["analyzer"],
            ngram_range=TITLE_CHAR_VECTORIZER_PARAMS["ngram_range"],
            max_features=TITLE_CHAR_VECTORIZER_PARAMS["max_features"],
            dtype=TITLE_CHAR_VECTORIZER_PARAMS["dtype"],
        )
    else:
        word_vectorizer = TfidfVectorizer(**TITLE_WORD_VECTORIZER_PARAMS)
        char_vectorizer = TfidfVectorizer(**TITLE_CHAR_VECTORIZER_PARAMS)
    word_features = word_vectorizer.fit_transform(texts)
    char_features = char_vectorizer.fit_transform(texts)
    if TITLE_CHAR_BLOCK_WEIGHT != 1.0:
        char_features = char_features.multiply(np.float32(TITLE_CHAR_BLOCK_WEIGHT))
    features = sparse.hstack(
        [word_features, char_features],
        format="csr",
        dtype=np.float32,
    )
    return word_vectorizer, char_vectorizer, features


def transform_title_feature_blocks(
    word_vectorizer: object,
    char_vectorizer: object,
    texts: list[str],
) -> sparse.csr_matrix:
    if word_vectorizer is None or char_vectorizer is None:
        return sparse.csr_matrix((len(texts), 0), dtype=np.float32)
    word_features = word_vectorizer.transform(texts)
    char_features = char_vectorizer.transform(texts)
    if TITLE_CHAR_BLOCK_WEIGHT != 1.0:
        char_features = char_features.multiply(np.float32(TITLE_CHAR_BLOCK_WEIGHT))
    return sparse.hstack(
        [word_features, char_features],
        format="csr",
        dtype=np.float32,
    )


def build_category_centroids(
    train_matrix: sparse.csr_matrix,
    train_categories: np.ndarray,
    category_department_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix, dict[int, np.ndarray]]:
    category_labels, inverse_indices = np.unique(
        train_categories,
        return_inverse=True,
    )
    row_indices = inverse_indices.astype(np.int32, copy=False)
    col_indices = np.arange(train_matrix.shape[0], dtype=np.int32)
    weights = np.ones(train_matrix.shape[0], dtype=np.float32)

    assignment = sparse.csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(len(category_labels), train_matrix.shape[0]),
        dtype=np.float32,
    )
    counts = np.asarray(assignment.sum(axis=1)).ravel()
    normalizer = sparse.diags(
        1.0 / np.maximum(counts, 1.0),
        dtype=np.float32,
    )
    category_centroid_matrix = (normalizer @ assignment @ train_matrix).tocsr()
    category_departments = np.array(
        [category_department_map[int(category)] for category in category_labels],
        dtype=np.int16,
    )
    department_category_indices = {
        int(department): np.flatnonzero(category_departments == department).astype(
            np.int32
        )
        for department in np.unique(category_departments)
    }
    return (
        category_labels.astype(np.int32, copy=False),
        category_departments,
        category_centroid_matrix,
        department_category_indices,
    )


def build_department_centroids(
    train_matrix: sparse.csr_matrix,
    train_departments: np.ndarray,
) -> tuple[np.ndarray, sparse.csr_matrix, dict[int, int]]:
    department_labels, inverse_indices = np.unique(
        train_departments,
        return_inverse=True,
    )
    row_indices = inverse_indices.astype(np.int32, copy=False)
    col_indices = np.arange(train_matrix.shape[0], dtype=np.int32)
    weights = np.ones(train_matrix.shape[0], dtype=np.float32)
    assignment = sparse.csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(len(department_labels), train_matrix.shape[0]),
        dtype=np.float32,
    )
    counts = np.asarray(assignment.sum(axis=1)).ravel()
    normalizer = sparse.diags(
        1.0 / np.maximum(counts, 1.0),
        dtype=np.float32,
    )
    centroid_matrix = (normalizer @ assignment @ train_matrix).tocsr()
    label_to_row = {int(label): int(i) for i, label in enumerate(department_labels)}
    return department_labels.astype(np.int64, copy=False), centroid_matrix, label_to_row


def fit_dense_bundle(
    dataframe: pd.DataFrame,
    category_labels: np.ndarray,
    department_labels: np.ndarray,
) -> dict[str, object]:
    dense_texts = build_dense_texts(dataframe)
    passage_texts = [DENSE_PASSAGE_PREFIX + text for text in dense_texts]
    dense_artifact = build_pruned_dense_artifact(
        model_name_or_path=DENSE_BACKBONE_PATH,
        train_passage_texts=passage_texts,
        num_hidden_layers=DENSE_NUM_HIDDEN_LAYERS,
        extra_base_token_count=DENSE_EXTRA_BASE_TOKEN_COUNT,
    )

    dense_embeddings = encode_texts_with_artifact(
        dense_artifact,
        dense_texts,
        prefix=DENSE_PASSAGE_PREFIX,
        max_length=DENSE_MAX_LENGTH,
        batch_size=DENSE_BATCH_SIZE,
    )

    dense_category_labels, dense_category_centroids = build_dense_centroids(
        dense_embeddings,
        dataframe["category_id"].to_numpy(dtype=np.int32, copy=False),
    )
    if not np.array_equal(category_labels, dense_category_labels):
        raise ValueError("Dense category labels are inconsistent with sparse labels")

    dense_department_labels, dense_department_centroids = build_dense_centroids(
        dense_embeddings,
        dataframe["department_id"].to_numpy(dtype=np.int64, copy=False),
    )
    if not np.array_equal(department_labels, dense_department_labels):
        raise ValueError("Dense department labels are inconsistent with sparse labels")

    return {
        "artifact": dense_artifact,
        "category_centroids": dense_category_centroids.astype(np.float16, copy=False),
        "department_centroids": dense_department_centroids.astype(np.float16, copy=False),
        "department_labels": dense_department_labels.astype(np.int64, copy=False),
    }


def fit_bundle(dataframe: pd.DataFrame) -> dict[str, object]:
    texts = build_search_texts(dataframe)
    title_texts = build_title_texts(dataframe)
    word_vectorizer, char_vectorizer, train_matrix = fit_feature_blocks(texts)
    title_word_vectorizer, title_char_vectorizer, title_matrix = fit_title_feature_blocks(
        title_texts
    )
    department_model = LinearSVC(**DEPARTMENT_MODEL_PARAMS)
    department_model.fit(train_matrix, dataframe["department_id"])

    train_categories = dataframe["category_id"].to_numpy(dtype=np.int32)
    train_departments = dataframe["department_id"].to_numpy(dtype=np.int64, copy=False)
    department_labels = np.unique(train_departments).astype(np.int64, copy=False)
    category_department_map = (
        dataframe.drop_duplicates("category_id")
        .set_index("category_id")["department_id"]
        .astype(np.int64)
        .to_dict()
    )
    (
        category_labels,
        category_departments,
        category_centroid_matrix,
        department_category_indices,
    ) = build_category_centroids(
        train_matrix,
        train_categories,
        category_department_map,
    )
    (
        title_category_labels,
        _,
        category_title_centroid_matrix,
        _,
    ) = build_category_centroids(
        title_matrix,
        train_categories,
        category_department_map,
    )
    if not np.array_equal(category_labels, title_category_labels):
        raise ValueError("Category labels are inconsistent across feature views")

    if CATEGORY_CENTROID_DEPT_SHRINK_ALPHA:
        alpha = float(CATEGORY_CENTROID_DEPT_SHRINK_ALPHA)
        if not (0.0 < alpha < 1.0):
            raise ValueError("CATEGORY_CENTROID_DEPT_SHRINK_ALPHA must be in (0, 1)")
        _, department_centroid_matrix, department_label_to_row = build_department_centroids(
            train_matrix,
            train_departments,
        )
        _, department_title_centroid_matrix, _ = build_department_centroids(
            title_matrix,
            train_departments,
        )
        dept_rows = np.array(
            [department_label_to_row[int(dept)] for dept in category_departments],
            dtype=np.int32,
        )
        dept_centroids_for_cat = department_centroid_matrix[dept_rows]
        category_centroid_matrix = category_centroid_matrix.multiply(
            np.float32(1.0 - alpha)
        ) + dept_centroids_for_cat.multiply(np.float32(alpha))

        title_dept_centroids_for_cat = department_title_centroid_matrix[dept_rows]
        category_title_centroid_matrix = category_title_centroid_matrix.multiply(
            np.float32(1.0 - alpha)
        ) + title_dept_centroids_for_cat.multiply(np.float32(alpha))

    override_maps = []
    for columns in OVERRIDE_RULES:
        mapping = build_unique_category_map(dataframe, columns)
        if columns in OVERRIDE_MAJORITY_RULES:
            majority = build_majority_category_map(dataframe, columns)
            if majority:
                majority.update(mapping)
                mapping = majority
        override_maps.append({"columns": columns, "mapping": mapping})

    normalized_override_maps: list[dict[str, object]] = []
    if USE_NORMALIZED_UNIQUE_OVERRIDES:
        vc_map = build_normalized_unique_category_map(
            dataframe, "vendor_code", clean_vendor_code
        )
        if vc_map:
            normalized_override_maps.append(
                {"column": "vendor_code", "mapping": vc_map, "normalizer": "vendor_code"}
            )
        title_map = build_normalized_unique_category_map(
            dataframe, "title", canonicalize_title
        )
        if title_map:
            normalized_override_maps.append(
                {"column": "title", "mapping": title_map, "normalizer": "title"}
            )

    vendor_code_norm_unique_map: dict[str, int] = {}
    vendor_code_norm_candidates_map: dict[str, np.ndarray] = {}
    if USE_VENDOR_CODE_NORM_DISAMBIGUATION:
        vendor_code_norm_unique_map = build_normalized_unique_category_map(
            dataframe, "vendor_code", clean_vendor_code
        )
        vendor_code_norm_candidates_map = build_normalized_candidates_map(
            dataframe,
            "vendor_code",
            clean_vendor_code,
            max_candidates=VENDOR_CODE_NORM_MAX_CANDIDATES,
        )

    dense_bundle = fit_dense_bundle(
        dataframe,
        category_labels,
        department_labels,
    )
    if not np.array_equal(
        dense_bundle["department_labels"],
        department_model.classes_.astype(np.int64, copy=False),
    ):
        raise ValueError("Dense department labels are inconsistent with department model")

    return {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "title_word_vectorizer": title_word_vectorizer,
        "title_char_vectorizer": title_char_vectorizer,
        "department_model": department_model,
        "category_labels": category_labels,
        "category_departments": category_departments,
        "category_centroid_matrix": category_centroid_matrix,
        "category_title_centroid_matrix": category_title_centroid_matrix,
        "department_category_indices": department_category_indices,
        "category_department_map": category_department_map,
        "override_maps": override_maps,
        "normalized_override_maps": normalized_override_maps,
        "vendor_code_norm_unique_map": vendor_code_norm_unique_map,
        "vendor_code_norm_candidates_map": vendor_code_norm_candidates_map,
        "dense_bundle": dense_bundle,
    }


def compute_sparse_similarity_scores(
    query_matrix: sparse.csr_matrix,
    category_centroid_matrix: sparse.csr_matrix,
    query_title_matrix: sparse.csr_matrix,
    category_title_centroid_matrix: sparse.csr_matrix,
) -> np.ndarray:
    scores = query_matrix @ category_centroid_matrix.T
    title_scores = query_title_matrix @ category_title_centroid_matrix.T
    return np.asarray(
        (scores + TITLE_BONUS_WEIGHT * title_scores).toarray(),
        dtype=np.float32,
    )


def compute_dense_similarity_scores(
    query_dense_embeddings: np.ndarray,
    category_dense_centroids: np.ndarray,
) -> np.ndarray:
    query = np.ascontiguousarray(query_dense_embeddings, dtype=np.float32)
    centroids = np.ascontiguousarray(category_dense_centroids, dtype=np.float32)
    return np.einsum("id,jd->ij", query, centroids, optimize=True).astype(
        np.float32,
        copy=False,
    )


def compute_category_similarity_scores(
    query_matrix: sparse.csr_matrix,
    category_centroid_matrix: sparse.csr_matrix,
    query_title_matrix: sparse.csr_matrix,
    category_title_centroid_matrix: sparse.csr_matrix,
    query_dense_embeddings: np.ndarray | None,
    category_dense_centroids: np.ndarray | None,
) -> np.ndarray:
    scores = compute_sparse_similarity_scores(
        query_matrix,
        category_centroid_matrix,
        query_title_matrix,
        category_title_centroid_matrix,
    )
    if (
        query_dense_embeddings is not None
        and category_dense_centroids is not None
        and DENSE_CATEGORY_SCORE_WEIGHT
    ):
        dense_scores = compute_dense_similarity_scores(
            query_dense_embeddings,
            category_dense_centroids,
        )
        scores += np.float32(DENSE_CATEGORY_SCORE_WEIGHT) * dense_scores.astype(
            np.float32, copy=False
        )
    return scores


def nearest_centroid_category_indices(
    query_matrix: sparse.csr_matrix,
    category_centroid_matrix: sparse.csr_matrix,
    query_title_matrix: sparse.csr_matrix,
    category_title_centroid_matrix: sparse.csr_matrix,
    query_dense_embeddings: np.ndarray | None,
    category_dense_centroids: np.ndarray | None,
    department_category_indices: dict[int, np.ndarray],
    top_departments: np.ndarray,
    top_department_scores: np.ndarray,
    uncertain_mask: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    predictions = np.empty(query_matrix.shape[0], dtype=np.int32)
    all_indices = np.arange(category_centroid_matrix.shape[0], dtype=np.int32)

    predicted_departments = top_departments[:, 0]
    confident_query_indices = np.flatnonzero(~uncertain_mask)

    for department in np.unique(predicted_departments[confident_query_indices]):
        query_indices = confident_query_indices[
            predicted_departments[confident_query_indices] == department
        ]
        if len(query_indices) == 0:
            continue

        candidate_indices = department_category_indices.get(
            int(department),
            all_indices,
        )
        candidate_matrix = category_centroid_matrix[candidate_indices]
        candidate_title_matrix = category_title_centroid_matrix[candidate_indices]

        for start in range(0, len(query_indices), batch_size):
            stop = min(start + batch_size, len(query_indices))
            batch_query_indices = query_indices[start:stop]
            dense_batch = None
            if query_dense_embeddings is not None:
                dense_batch = query_dense_embeddings[batch_query_indices]
            dense_candidates = None
            if category_dense_centroids is not None:
                dense_candidates = category_dense_centroids[candidate_indices]
            scores = compute_category_similarity_scores(
                query_matrix[batch_query_indices],
                candidate_matrix,
                query_title_matrix[batch_query_indices],
                candidate_title_matrix,
                dense_batch,
                dense_candidates,
            )
            best_local_indices = np.asarray(scores.argmax(axis=1)).ravel()
            predictions[batch_query_indices] = candidate_indices[best_local_indices]

    for query_index in np.flatnonzero(uncertain_mask):
        best_adjusted_score = -np.inf
        best_category_index = -1
        for dept_pos, department in enumerate(top_departments[query_index]):
            candidate_indices = department_category_indices.get(
                int(department),
                all_indices,
            )
            if len(candidate_indices) == 0:
                candidate_indices = all_indices
            candidate_matrix = category_centroid_matrix[candidate_indices]
            candidate_title_matrix = category_title_centroid_matrix[candidate_indices]
            dense_query = None
            if query_dense_embeddings is not None:
                dense_query = query_dense_embeddings[query_index : query_index + 1]
            dense_candidates = None
            if category_dense_centroids is not None:
                dense_candidates = category_dense_centroids[candidate_indices]
            scores = compute_category_similarity_scores(
                query_matrix[query_index : query_index + 1],
                candidate_matrix,
                query_title_matrix[query_index : query_index + 1],
                candidate_title_matrix,
                dense_query,
                dense_candidates,
            )
            best_local_index = int(np.asarray(scores.argmax(axis=1)).ravel()[0])
            best_score = float(np.asarray(scores.max()).ravel()[0])
            dept_score = float(top_department_scores[query_index, dept_pos])
            adjusted = best_score + DEPARTMENT_SCORE_BONUS_WEIGHT_FOR_CATEGORY * dept_score
            if adjusted > best_adjusted_score:
                best_adjusted_score = adjusted
                best_category_index = int(candidate_indices[best_local_index])

        if best_category_index < 0:
            best_category_index = int(all_indices[0])
        predictions[query_index] = best_category_index
    return predictions


def should_sync_department_to_category(
    query_features: sparse.csr_matrix,
    query_title_features: sparse.csr_matrix,
    query_dense_embeddings: np.ndarray | None,
    selected_category_index: int,
    top_department: int,
    model_bundle: dict[str, object],
) -> bool:
    top_department_indices = model_bundle["department_category_indices"].get(
        int(top_department),
        np.empty(0, dtype=np.int32),
    )
    if len(top_department_indices) == 0:
        return False

    dense_query = None
    if query_dense_embeddings is not None:
        dense_query = query_dense_embeddings

    dense_category_centroids = (
        model_bundle["dense_bundle"]["category_centroids"].astype(np.float32, copy=False)
        if model_bundle.get("dense_bundle") is not None
        else None
    )
    selected_scores = compute_category_similarity_scores(
        query_features,
        model_bundle["category_centroid_matrix"][
            selected_category_index : selected_category_index + 1
        ],
        query_title_features,
        model_bundle["category_title_centroid_matrix"][
            selected_category_index : selected_category_index + 1
        ],
        dense_query,
        None
        if dense_category_centroids is None
        else dense_category_centroids[
            selected_category_index : selected_category_index + 1
        ],
    )
    top_department_scores = compute_category_similarity_scores(
        query_features,
        model_bundle["category_centroid_matrix"][top_department_indices],
        query_title_features,
        model_bundle["category_title_centroid_matrix"][top_department_indices],
        dense_query,
        None
        if dense_category_centroids is None
        else dense_category_centroids[top_department_indices],
    )
    selected_score = float(np.asarray(selected_scores).ravel()[0])
    best_top_department_score = float(np.asarray(top_department_scores.max()).ravel()[0])
    return (
        selected_score - best_top_department_score
    ) >= DEPARTMENT_SYNC_SCORE_GAP_THRESHOLD


def apply_category_overrides_with_flags(
    dataframe: pd.DataFrame,
    base_categories: np.ndarray,
    override_maps: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = base_categories.copy()
    override_used = np.zeros(len(predictions), dtype=bool)
    if len(predictions) == 0:
        return predictions, override_used

    rule_keys = []
    for rule in override_maps:
        columns = rule["columns"]
        rule_keys.append(build_rule_keys(dataframe, columns))

    for row_index in range(len(predictions)):
        for rule_index, rule in enumerate(override_maps):
            mapping = rule["mapping"]
            key = rule_keys[rule_index][row_index]
            if has_meaningful_key(key) and key in mapping:
                predictions[row_index] = mapping[key]
                override_used[row_index] = True
                break
    return predictions, override_used


def apply_normalized_unique_overrides(
    dataframe: pd.DataFrame,
    predictions: np.ndarray,
    override_used: np.ndarray,
    normalized_override_maps: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    if not normalized_override_maps:
        return predictions, override_used
    if len(predictions) == 0:
        return predictions, override_used

    remaining = np.flatnonzero(~override_used)
    if len(remaining) == 0:
        return predictions, override_used

    for rule in normalized_override_maps:
        column = str(rule.get("column", ""))
        normalizer = str(rule.get("normalizer", ""))
        mapping: dict[str, int] = rule.get("mapping", {})
        if not column or not mapping:
            continue

        if normalizer == "vendor_code":
            keys = dataframe[column].map(clean_vendor_code).to_numpy(dtype=object)
        elif normalizer == "title":
            keys = dataframe[column].map(canonicalize_title).to_numpy(dtype=object)
        else:
            continue

        for row_index in remaining:
            key = keys[row_index]
            if has_meaningful_key(key) and key in mapping:
                predictions[row_index] = mapping[key]
                override_used[row_index] = True

        remaining = np.flatnonzero(~override_used)
        if len(remaining) == 0:
            break

    return predictions, override_used


def apply_category_overrides(
    dataframe: pd.DataFrame,
    base_categories: np.ndarray,
    override_maps: list[dict[str, object]],
) -> np.ndarray:
    predictions, _ = apply_category_overrides_with_flags(
        dataframe,
        base_categories,
        override_maps,
    )
    return predictions


def get_department_candidates(
    department_model: LinearSVC,
    query_matrix: sparse.csr_matrix,
    query_dense_embeddings: np.ndarray | None,
    dense_department_labels: np.ndarray | None,
    dense_department_centroids: np.ndarray | None,
    top_k: int = UNCERTAIN_TOP_DEPARTMENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decision_scores = department_model.decision_function(query_matrix)
    if decision_scores.ndim == 1:
        decision_scores = np.vstack([-decision_scores, decision_scores]).T

    if (
        query_dense_embeddings is not None
        and dense_department_centroids is not None
        and dense_department_labels is not None
        and DENSE_DEPARTMENT_SCORE_WEIGHT
    ):
        if not np.array_equal(
            department_model.classes_.astype(np.int64, copy=False),
            dense_department_labels.astype(np.int64, copy=False),
        ):
            raise ValueError("Dense department labels do not match department model classes")
        dense_scores = compute_dense_similarity_scores(
            query_dense_embeddings,
            dense_department_centroids,
        )
        decision_scores = decision_scores + np.float32(
            DENSE_DEPARTMENT_SCORE_WEIGHT
        ) * dense_scores.astype(np.float32, copy=False)

    n_samples, n_classes = decision_scores.shape
    top_k = max(1, min(int(top_k), int(n_classes)))
    if top_k == 1:
        top_indices = np.argmax(decision_scores, axis=1)[:, None]
        top_scores = decision_scores[np.arange(n_samples), top_indices[:, 0]][:, None]
    else:
        top_indices = np.argpartition(-decision_scores, kth=top_k - 1, axis=1)[:, :top_k]
        top_scores = decision_scores[np.arange(n_samples)[:, None], top_indices]
        order = np.argsort(-top_scores, axis=1)
        top_indices = top_indices[np.arange(n_samples)[:, None], order]
        top_scores = top_scores[np.arange(n_samples)[:, None], order]

    top_departments = np.asarray(department_model.classes_[top_indices], dtype=np.int64)
    if top_k >= 2:
        uncertain_mask = (top_scores[:, 0] - top_scores[:, 1]) < UNCERTAIN_MARGIN_THRESHOLD
    else:
        uncertain_mask = np.zeros(n_samples, dtype=bool)
    return top_departments, top_scores.astype(np.float32, copy=False), uncertain_mask


def predict_with_bundle(
    model_bundle: dict[str, object], dataframe: pd.DataFrame
) -> pd.DataFrame:
    texts = build_search_texts(dataframe)
    title_texts = build_title_texts(dataframe)
    dense_texts = build_dense_texts(dataframe)
    query_matrix = transform_feature_blocks(
        model_bundle["word_vectorizer"],
        model_bundle["char_vectorizer"],
        texts,
    )
    query_title_matrix = transform_title_feature_blocks(
        model_bundle["title_word_vectorizer"],
        model_bundle["title_char_vectorizer"],
        title_texts,
    )
    dense_query_embeddings = encode_texts_with_artifact(
        model_bundle["dense_bundle"]["artifact"],
        dense_texts,
        prefix=DENSE_QUERY_PREFIX,
        max_length=DENSE_MAX_LENGTH,
        batch_size=DENSE_BATCH_SIZE,
    )
    dense_category_centroids = model_bundle["dense_bundle"]["category_centroids"].astype(
        np.float32,
        copy=False,
    )
    dense_department_centroids = model_bundle["dense_bundle"][
        "department_centroids"
    ].astype(np.float32, copy=False)

    top_departments, top_department_scores, uncertain_mask = get_department_candidates(
        model_bundle["department_model"],
        query_matrix,
        dense_query_embeddings,
        model_bundle["dense_bundle"]["department_labels"],
        dense_department_centroids,
    )
    department_predictions = top_departments[:, 0]
    base_category_indices = nearest_centroid_category_indices(
        query_matrix,
        model_bundle["category_centroid_matrix"],
        query_title_matrix,
        model_bundle["category_title_centroid_matrix"],
        dense_query_embeddings,
        dense_category_centroids,
        model_bundle["department_category_indices"],
        top_departments,
        top_department_scores,
        uncertain_mask,
    )
    base_categories = model_bundle["category_labels"][base_category_indices]
    category_predictions, override_used = apply_category_overrides_with_flags(
        dataframe,
        base_categories,
        model_bundle["override_maps"],
    )
    category_predictions, override_used = apply_normalized_unique_overrides(
        dataframe,
        category_predictions,
        override_used,
        model_bundle.get("normalized_override_maps", []),
    )
    if USE_VENDOR_CODE_NORM_DISAMBIGUATION:
        vc_unique = model_bundle.get("vendor_code_norm_unique_map") or {}
        vc_candidates = model_bundle.get("vendor_code_norm_candidates_map") or {}
        if vc_unique or vc_candidates:
            vc_keys = dataframe["vendor_code"].map(clean_vendor_code).to_numpy(dtype=object)
            category_labels = model_bundle["category_labels"]
            for row_index in np.flatnonzero(~override_used):
                key = vc_keys[row_index]
                if not has_meaningful_key(key):
                    continue
                if key in vc_unique:
                    category_predictions[row_index] = int(vc_unique[key])
                    override_used[row_index] = True
                    continue
                candidates = vc_candidates.get(str(key))
                if candidates is None:
                    continue
                if len(candidates) == 0:
                    continue
                cand_indices = np.searchsorted(category_labels, candidates).astype(
                    np.int32, copy=False
                )
                cand_arr = compute_category_similarity_scores(
                    query_matrix[row_index : row_index + 1],
                    model_bundle["category_centroid_matrix"][cand_indices],
                    query_title_matrix[row_index : row_index + 1],
                    model_bundle["category_title_centroid_matrix"][cand_indices],
                    dense_query_embeddings[row_index : row_index + 1],
                    dense_category_centroids[cand_indices],
                )
                cand_arr = np.asarray(cand_arr).ravel().astype(np.float32, copy=False)
                best_pos = int(cand_arr.argmax())
                best_score = float(cand_arr[best_pos])
                base_score_mat = compute_category_similarity_scores(
                    query_matrix[row_index : row_index + 1],
                    model_bundle["category_centroid_matrix"][
                        int(base_category_indices[row_index]) : int(base_category_indices[row_index]) + 1
                    ],
                    query_title_matrix[row_index : row_index + 1],
                    model_bundle["category_title_centroid_matrix"][
                        int(base_category_indices[row_index]) : int(base_category_indices[row_index]) + 1
                    ],
                    dense_query_embeddings[row_index : row_index + 1],
                    dense_category_centroids[
                        int(base_category_indices[row_index]) : int(base_category_indices[row_index]) + 1
                    ],
                )
                base_score = float(np.asarray(base_score_mat).ravel()[0])
                if (best_score - base_score) < float(VENDOR_CODE_NORM_MIN_GAIN):
                    continue
                category_predictions[row_index] = int(category_labels[int(cand_indices[best_pos])])
                override_used[row_index] = True
    final_departments = department_predictions.astype(np.int64, copy=True)
    base_category_departments = model_bundle["category_departments"][
        base_category_indices
    ].astype(np.int64, copy=False)
    category_department_map = model_bundle["category_department_map"]
    sync_candidate_indices = np.flatnonzero(
        (~override_used) & (base_category_departments != final_departments)
    )
    for index in sync_candidate_indices:
        if should_sync_department_to_category(
            query_matrix[index : index + 1],
            query_title_matrix[index : index + 1],
            dense_query_embeddings[index : index + 1],
            int(base_category_indices[index]),
            int(department_predictions[index]),
            model_bundle,
        ):
            final_departments[index] = base_category_departments[index]

    override_indices = np.flatnonzero(override_used)
    if len(override_indices) > 0:
        final_departments[override_indices] = np.array(
            [
                category_department_map.get(
                    int(category_predictions[index]),
                    int(final_departments[index]),
                )
                for index in override_indices
            ],
            dtype=np.int64,
        )

    return pd.DataFrame(
        {
            "department_id": final_departments,
            "category_id": category_predictions.astype(np.int64),
        }
    )


def score_predictions(
    truth: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, float]:
    department_f1 = f1_score(
        truth["department_id"],
        predictions["department_id"],
        average="macro",
    )
    category_accuracy = accuracy_score(
        truth["category_id"],
        predictions["category_id"],
    )
    score = 30.0 * department_f1 + 70.0 * category_accuracy
    return {
        "score": score,
        "department_f1": department_f1,
        "category_accuracy": category_accuracy,
    }

def evaluate_with_splits(
        dataframe: pd.DataFrame,
        n_splits: int = 3,
        test_size: float = 0.2,
        random_state: int = 42,
) -> dict[str, float]:

    cat_to_indices: dict[int, list[int]] = {}
    for idx, cat in zip(dataframe.index, dataframe["category_id"]):
        cat_to_indices.setdefault(int(cat), []).append(idx)

    triple_cats = [c for c, idxs in cat_to_indices.items() if len(idxs) == 3]
    double_cats = [c for c, idxs in cat_to_indices.items() if len(idxs) == 2]
    single_cats = [c for c, idxs in cat_to_indices.items() if len(idxs) == 1]

    target_val = int(test_size * len(dataframe))
    n_from_triples = len(triple_cats)
    n_from_doubles = max(0, min(target_val - n_from_triples, len(double_cats)))

    scores = []

    for fold_idx in range(n_splits):
        rng = np.random.RandomState(random_state + fold_idx)

        val_indices: list[int] = []
        train_indices: list[int] = []

        # 1) Все triple-категории: 1 рандомная запись в val, 2 в train
        for cat_id in triple_cats:
            indices = list(cat_to_indices[cat_id])
            rng.shuffle(indices)
            val_indices.append(indices[0])
            train_indices.extend(indices[1:])

        # 2) Рандомная выборка из double-категорий
        selected_doubles = set(
            rng.choice(double_cats, size=n_from_doubles, replace=False)
        )
        for cat_id in double_cats:
            indices = list(cat_to_indices[cat_id])
            if cat_id in selected_doubles:
                rng.shuffle(indices)
                val_indices.append(indices[0])
                train_indices.extend(indices[1:])
            else:
                train_indices.extend(indices)

        # 3) Single-категории: только train
        for cat_id in single_cats:
            train_indices.extend(cat_to_indices[cat_id])

        train_fold = dataframe.loc[train_indices].reset_index(drop=True)
        val_fold = dataframe.loc[val_indices].reset_index(drop=True)

        # Проверка: orphan categories
        val_cats = set(val_fold["category_id"].unique())
        train_cats = set(train_fold["category_id"].unique())
        orphan = val_cats - train_cats

        print(f"\n--- Fold {fold_idx + 1}/{n_splits} "
              f"(train={len(train_fold)}, val={len(val_fold)}, "
              f"orphan_cats={len(orphan)}) ---")

        model_bundle = fit_bundle(train_fold)
        predictions = predict_with_bundle(model_bundle, val_fold)
        fold_score = score_predictions(val_fold, predictions)
        scores.append(fold_score)

        print(f"  Score={fold_score['score']:.4f}  "
              f"F1_dept={fold_score['department_f1']:.4f}  "
              f"Acc_cat={fold_score['category_accuracy']:.4f}  ")

    metrics = {
        "score_mean": float(np.mean([s["score"] for s in scores])),
        "score_std": float(np.std([s["score"] for s in scores])),
        "department_f1_mean": float(np.mean([s["department_f1"] for s in scores])),
        "category_accuracy_mean": float(np.mean([s["category_accuracy"] for s in scores])),
    }

    print(f"\n  >>> MEAN Score: {metrics['score_mean']:.4f} "
          f"± {metrics['score_std']:.4f}")
    print(f"  >>> MEAN F1_dept:  {metrics['department_f1_mean']:.4f}")
    print(f"  >>> MEAN Acc_cat:  {metrics['category_accuracy_mean']:.4f}")
    return metrics


def save_bundle(model_bundle: dict[str, object], path: str | Path) -> None:
    with gzip.open(path, "wb") as file:
        pickle.dump(model_bundle, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path: str | Path) -> dict[str, object]:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def predict_to_csv(
    model_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    model_bundle = load_bundle(model_path)
    test = load_dataset(test_path)
    predictions = predict_with_bundle(model_bundle, test)
    predictions.to_csv(output_path, index=False)
    return predictions


if __name__ == "__main__":
    df = load_dataset("train.tsv")
    print(evaluate_with_splits(df))
