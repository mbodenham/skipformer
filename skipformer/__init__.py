from typing import TYPE_CHECKING
from transformers.utils import (
    # OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)

_import_structure = {
    "configuration_skipformer": ["SKIPFORMER_PRETRAINED_CONFIG", "SkipformerConfig"],
    "tokenization_skipformer": ["SkipformerTokenizer"],
    "trainer": ["SkipformerTrainer", "LogWindowSize"],
    "args": ['parser']
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_skipformer_fast"] = ["SkipformerTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_skipformer"] = [
        "SKIPFORMERPRETRAINED_MODEL_ARCHIVE_LIST",
        # "GPT2DoubleHeadsModel",
        "SkipformerForSequenceClassification",
        # "GPT2ForTokenClassification",
        "SkipformerLMHeadModel",
        "SkipformerModel",
        "SkipformerPreTrainedModel",
        # "GPT2Attention"
    ]
    _import_structure['attention_window_cuda'] = ["attention_window_matmul"]

if TYPE_CHECKING:
    from .configuration_skipformer import SKIPFORMER_PRETRAINED_CONFIG, SkipformerConfig
    from .tokenization_skipformer import SkipformerTokenizer
    from .args import parser

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_skipformer_fast import SkipformerTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_skipformer import (
            SKIPFORMERPRETRAINED_MODEL_ARCHIVE_LIST,
            # GPT2DoubleHeadsModel,
            SkipformerForSequenceClassification,
            # GPT2ForTokenClassification,
            SkipformerLMHeadModel,
            SkipformerModel,
            SkipformerPreTrainedModel,
        )
        from .attention_window_cuda import attention_window_matmul


else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
