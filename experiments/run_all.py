from __future__ import annotations

from pathlib import Path
import json

from experiments.cases.case1_forward_dt_vbc import run as run_case1
from experiments.cases.case2_backward_dt_vbc import run as run_case2
from experiments.cases.case3_bibc_bridge import run as run_case3
from experiments.common import DATA


def main() -> None:
    base = Path(__file__).resolve().parent / 'cases' / 'configs'
    summary = {
        'case1_forward_dt_vbc': run_case1(base / 'case1_forward.yaml'),
        'case2_backward_dt_vbc': run_case2(base / 'case2_backward.yaml'),
        'case3_bibc_bridge': run_case3(base / 'case3_bibc.yaml'),
    }
    with open(DATA / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
