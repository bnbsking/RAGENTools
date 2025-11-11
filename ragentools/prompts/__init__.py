from typing import Dict, Type, Tuple, Union
import yaml

from pydantic import BaseModel
from ragentools.common.formatting import get_response_model


class ResponseFormatProcessor:
    def dummy(response_format: Dict) -> Dict:
        return response_format

    def model(response_format: Dict) -> Type:
        response_model = get_response_model(response_format)
        return response_model


def get_prompt_and_response_format(
        prompt_path: str,
        replace_dict: dict = {},
        response_process: str = "dummy"
    ) -> Tuple[str, Union[Dict, Type[BaseModel]]]:
    with open(prompt_path, 'r') as file:
        prompt_cfg = yaml.safe_load(file)
    prompt = prompt_cfg["prompt"]
    replace_dict = prompt_cfg["default_replacements"] | replace_dict
    
    for key, value in replace_dict.items():
        prompt = prompt.replace(f"{{{{ {key} }}}}", str(value))
    response_format = getattr(ResponseFormatProcessor, response_process)(prompt_cfg['response_format'])
    return prompt, response_format
