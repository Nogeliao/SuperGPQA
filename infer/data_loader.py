from utils.common import read_yaml, read_json_or_jsonl

# Logical split: same file as SuperGPQA-all, filtered to Engineering /
# "Computer Science and Technology" only (must match dataset JSON exactly).
# Usage: python infer/infer.py ... --split SuperGPQA-Engineering-CS --mode zero-shot ...
_SUPERGPQA_ENGINEERING_CS = "SuperGPQA-Engineering-CS"
_SUPERGPQA_SPLITS_WITH_MODES = ("SuperGPQA-all", _SUPERGPQA_ENGINEERING_CS)
_CS_FIELD = "Computer Science and Technology"


def _resolve_split(split):
    """Return (file_split, filter_fn) for read_json_or_jsonl."""
    if split == _SUPERGPQA_ENGINEERING_CS:
        def _filter(items):
            return [
                item
                for item in items
                if item.get("discipline") == "Engineering"
                and item.get("field") == _CS_FIELD
            ]

        return "SuperGPQA-all", _filter
    return split, lambda x: x


def load_data(split='', mode=''):
    if split in _SUPERGPQA_SPLITS_WITH_MODES and mode in ['zero-shot', 'zero-shot-bon', 'five-shot']:
        file_split, filter_fn = _resolve_split(split)
        sample = filter_fn(read_json_or_jsonl('data', file_split))
        config = mode.replace('-bon', '')
        template = read_yaml(config)
        for item in sample:
            prompt_format = [item['question']+'\n'+'\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(item['options'])])]
            prompt = template['prompt_format'][0].format(*prompt_format)
            yield prompt, item

    elif split in _SUPERGPQA_SPLITS_WITH_MODES and mode in ['zero-shot-with-subfield']:
        file_split, filter_fn = _resolve_split(split)
        sample = filter_fn(read_json_or_jsonl('data', file_split))
        config = 'zero-shot-with-subfield'
        template = read_yaml(config)
        for item in sample:
            prompt_format = [item['subfield'], item['question']+'\n'+'\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(item['options'])])]
            prompt = template['prompt_format'][0].format(*prompt_format)
            yield prompt, item

    elif split in _SUPERGPQA_SPLITS_WITH_MODES and 'robustness-exp' in mode:
        file_split, filter_fn = _resolve_split(split)
        sample = filter_fn(read_json_or_jsonl('data', file_split))
        config = 'robustness-exp'
        template = read_yaml(config)
        prompt_index, format_index = mode.split('-')[-2], mode.split('-')[-1]

        for item in sample:
            question_format_list = [
                item['question']+ '\n' + '\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(item['options'])]),
                item['question']+ '\n' + '\n'.join([f'{chr(65+i)}. {option}' for i, option in enumerate(item['options'])]) + '\n' + 'Your response: ',
                'Question: ' + item['question'] + '\n' + 'Options:\n' + '\n'.join([f'{chr(65+i)}: {option}' for i, option in enumerate(item['options'])]),
                'Question:\n' + item['question'] + '\n' + 'Options:\n' + '\n'.join([f'{chr(65+i)}) {option}' for i, option in enumerate(item['options'])]) + '\n' + 'Please begin answering.',
                'Q: ' + item['question'] + '\n' +' '.join([f'({chr(65+i)}) {option}' for i, option in enumerate(item['options'])]),
                '**Question**:\n' + item['question']+ '\n' + '**Options**:\n' + '\n'.join([f'({chr(65+i)}) {option}' for i, option in enumerate(item['options'])]),
            ]
            prompt = template[f'initial_prompt_{prompt_index}'][0].format(question_format_list[int(format_index)])
            yield prompt, item

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <mode>")
        sys.exit(1)

    mode = sys.argv[1]
    last_prompt = None
    from tqdm import tqdm
    for prompt, sample in tqdm(load_data('SuperGPQA-all', mode), desc='Loading data'):
        last_prompt = prompt
        last_sample = sample
        break

    if last_prompt is not None:
        print(last_prompt)
        print('-'*100)
