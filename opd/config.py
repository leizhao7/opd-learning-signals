"""Load experiment recipes and apply deterministic Hydra overrides."""
import json
import re

from .paths import ROOT

PLACEHOLDER = re.compile(r"\$\{([A-Z_]+)\}")


def recipe_names():
    return sorted(path.stem for path in (ROOT / 'configs/recipes').glob('*.json'))


def override_key(item):
    key, separator, _ = item.partition('=')
    if not separator or not key.lstrip('+').strip() or key != key.strip():
        raise ValueError(f'Expected a Hydra key=value assignment: {item!r}')
    return key.lstrip('+')


def merge_overrides(base, updates):
    """Last assignment wins, including replacement of +key with key."""
    result = {override_key(item): item for item in base}
    for item in updates:
        result[override_key(item)] = item
    return list(result.values())


def load_recipe(name):
    if name not in recipe_names():
        raise ValueError(f'Unknown recipe: {name!r}')
    recipe = json.loads((ROOT / 'configs/recipes' / f'{name}.json').read_text())
    base_name = recipe.get('extends')
    if base_name not in (None, 'base.json'):
        raise ValueError(f'Unsupported recipe base: {base_name!r}')
    base = json.loads((ROOT / 'configs/base.json').read_text())['overrides'] if base_name else []
    for source in (base, recipe['overrides']):
        keys = [override_key(item) for item in source]
        if len(keys) != len(set(keys)):
            raise ValueError(f'Duplicate assignments in {name}')
    recipe['overrides'] = merge_overrides(base, recipe['overrides'])
    vendor = ROOT / 'vendor' / recipe['vendor']
    if recipe['vendor'] not in {'verl-original', 'verl-code', 'verl-qwen', 'verl-full'}:
        raise ValueError(f'Unknown backend: {recipe["vendor"]}')
    if not (vendor / 'verl/version/version').is_file():
        raise ValueError(f'Missing backend: {vendor}')
    return recipe


def render_overrides(overrides, values):
    """Substitute known placeholders without interpreting Hydra values."""
    rendered = []
    for item in overrides:
        def replace(match):
            key = match.group(1)
            if key not in values:
                raise ValueError(f'Unknown placeholder: {key}')
            return values[key]
        rendered.append(PLACEHOLDER.sub(replace, item))
    return rendered
