import src.module as module_api
from src.module.configuration import QiChatConfig
from src.module.model import QiChatForCausalLM, QiChatModel
from src.module.model_presets import MODEL_PRESETS


def test_public_api_exports_expected_symbols() -> None:
    assert set(module_api.__all__) == {'QiChatModel', 'QiChatForCausalLM', 'QiChatConfig', 'MODEL_PRESETS'}
    assert module_api.QiChatModel is QiChatModel
    assert module_api.QiChatForCausalLM is QiChatForCausalLM
    assert module_api.QiChatConfig is QiChatConfig
    assert module_api.MODEL_PRESETS is MODEL_PRESETS
