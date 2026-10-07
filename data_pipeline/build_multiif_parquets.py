"""从 Multi-IF 源 CSV 构建三份 parquet，schema 与本项目 dapo-math-17k.parquet 完全一致。

  train_turn1.parquet   4501 行，第一轮 prompt，data_source=MultiIF（论文公开配方的 IF 训练集）
  val_turn1.parquet     同一批 prompt，data_source=MultiIF_<language>，供训练循环内按语言出单轮指标
  eval_3turn.parquet    4445 行三轮完整对话，只给离线评测器读
"""
import argparse, json
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
    ("ability", pa.string()),
    ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
    ("extra_info", pa.struct([("index", pa.string())])),
])


def parse_kwargs(raw):
    return [json.loads(k) if isinstance(k, str) else k for k in json.loads(raw)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.src, keep_default_na=False)
    train, val, ev = [], [], []
    for _, r in df.iterrows():
        turn1 = json.loads(r["turn_1_prompt"]); ids = json.loads(r["turn_1_instruction_id_list"]); kw = parse_kwargs(r["turn_1_kwargs"])
        gt = json.dumps({"key": str(r["key"]), "prompt": turn1["content"], "instruction_id_list": ids, "kwargs": kw}, ensure_ascii=False)
        base = {"prompt": [{"content": turn1["content"], "role": "user"}], "ability": "instruction_following",
                "reward_model": {"ground_truth": gt, "style": "rule"}, "extra_info": {"index": str(r["key"])}}
        train.append(dict(base, data_source="MultiIF"))
        val.append(dict(base, data_source=f"MultiIF_{r['language']}"))
        if r["turn_3_prompt"]:
            row = {"key": str(r["key"]), "language": str(r["language"])}
            for t in (1, 2, 3):
                row[f"turn_{t}_prompt"] = json.loads(r[f"turn_{t}_prompt"])["content"]
                row[f"turn_{t}_instruction_id_list"] = json.dumps(json.loads(r[f"turn_{t}_instruction_id_list"]))
                row[f"turn_{t}_kwargs"] = json.dumps(parse_kwargs(r[f"turn_{t}_kwargs"]), ensure_ascii=False)
            ev.append(row)
    for name, rows in [("train_turn1", train), ("val_turn1", val)]:
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out / f"{name}.parquet")
    pd.DataFrame(ev).to_parquet(out / "eval_3turn.parquet", index=False)
    print({"train_turn1": len(train), "val_turn1": len(val), "eval_3turn": len(ev)})
    print(pd.Series([v["data_source"] for v in val]).value_counts().to_dict())


if __name__ == "__main__":
    main()
