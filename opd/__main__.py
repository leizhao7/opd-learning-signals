"""Unified, dependency-light command line for the OPD release."""
import subprocess
import sys

from .paths import ROOT

# Scientific implementations keep their original CLI and optional dependencies.
TOOLS = {
    'sft': ('training/sft.py', 'Run SFT using a resolved JSON configuration'),
    'signals': ('analysis/learning_signal.py', 'Compute diagnostic rate factors from CSV'),
    'code-audit': ('evaluation/audit_code_avg4.py', 'Audit fixed avg@4 code evaluation'),
    'cka': ('analysis/geometry/compute_cka.py', 'Compute representation similarity'),
    'weights': ('analysis/geometry/compute_weight_metrics.py', 'Compute parameter distances'),
    'representations': ('analysis/geometry/extract_representations.py', 'Extract model representations'),
    'input-banks': ('analysis/geometry/build_input_banks.py', 'Build geometry input banks'),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h', '--help'):
        print('Usage: python -m opd COMMAND [OPTIONS]\n')
        for name, description in [('train', 'Render or execute a training recipe'),
                                  ('figures', 'Reproduce all or selected figures'),
                                  ('recipes', 'Validate and list available recipes'),
                                  *[(k, v[1]) for k, v in TOOLS.items()]]:
            print(f'  {name:18} {description}')
        return
    command, *rest = args
    if command == 'train':
        from .train import main as run
        run(rest)
    elif command == 'figures':
        from .figures import main as run
        run(rest)
    elif command == 'recipes':
        if rest:
            raise SystemExit('Usage: python -m opd recipes')
        from .config import load_recipe, recipe_names
        for name in recipe_names():
            recipe = load_recipe(name)
            print(f'{name:20} {recipe["vendor"]:16} {len(recipe["overrides"]):3} settings')
    elif command in TOOLS:
        result = subprocess.run([sys.executable, str(ROOT / TOOLS[command][0]), *rest])
        raise SystemExit(result.returncode)
    else:
        raise SystemExit(f'Unknown command {command!r}. Run python -m opd --help.')


if __name__ == '__main__':
    main()
