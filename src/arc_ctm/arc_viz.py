"""Dependency-free HTML rendering for ARC task predictions."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

from arc_ctm.arc_data import ArcTask


ARC_COLORS = (
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
)


def _grid(grid: Sequence[Sequence[int]], label: str) -> str:
    rows = []
    for row in grid:
        cells = "".join(
            f'<span class="cell" style="background:{ARC_COLORS[value]}" '
            f'aria-label="color {value}"><b>{value}</b></span>'
            for value in row
        )
        rows.append(f'<div class="grid-row">{cells}</div>')
    return (
        f'<div class="grid" role="img" aria-label="{html.escape(label)}">'
        + "".join(rows)
        + "</div>"
    )


def write_arc_report(task: ArcTask, results: dict[str, Any], path: Path) -> None:
    """Write a standalone visual report for one ARC overfit run."""

    demonstrations = []
    for index, pair in enumerate(task.train, start=1):
        demonstrations.append(
            '<section class="pair">'
            f"<h3>Demonstration {index}</h3>"
            '<div class="grid-group">'
            f'<div><span>Input</span>{_grid(pair.input.tolist(), "demonstration input")}</div>'
            f'<div><span>Output</span>{_grid(pair.output.tolist(), "demonstration output")}</div>'
            "</div></section>"
        )

    test = task.test[0]
    comparison = [
        ("Test input", test.input.tolist()),
        ("Expected", test.output.tolist()),
        ("Untrained", results["untrained"]["final_prediction"]),
        ("After training", results["trained"]["final_prediction"]),
        ("Corrupted support", results["corrupted_support"]["final_prediction"]),
    ]
    comparison_html = "".join(
        f'<div><span>{html.escape(label)}</span>{_grid(grid, label)}</div>'
        for label, grid in comparison
    )
    status = "CORRECT" if results["trained"]["exact"] else "INCORRECT"
    payload = html.escape(json.dumps(results["metrics"], indent=2))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARC task {html.escape(task.task_id)} overfit report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#17202a}}main{{max-width:1100px;margin:auto}}
h1,h2,h3{{font-weight:600}}.summary{{padding:12px 16px;background:white;border:1px solid #d8dee4;border-radius:8px}}
.demos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}}.pair{{background:white;padding:12px;border-radius:8px}}
.grid-group,.comparison{{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}}.grid-group>div>span,.comparison>div>span{{display:block;margin-bottom:6px;font-weight:600}}
.grid{{display:inline-block;border:2px solid #45525f;background:#45525f;line-height:0}}.grid-row{{display:flex}}
.cell{{width:30px;height:30px;margin:1px;display:inline-flex;align-items:center;justify-content:center;color:white;font-size:11px;line-height:1;text-shadow:0 1px 2px #000}}
pre{{background:#17202a;color:#eaf2f8;padding:12px;border-radius:8px;overflow:auto}}
</style></head><body><main>
<h1>ARC-AGI-1 task {html.escape(task.task_id)}</h1>
<p class="summary">Held-out official test prediction: <strong>{status}</strong>. The test target was used only for final evaluation.</p>
<h2>Training demonstrations</h2><div class="demos">{''.join(demonstrations)}</div>
<h2>Official test comparison</h2><div class="comparison">{comparison_html}</div>
<h2>Metrics</h2><pre>{payload}</pre>
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_multitask_report(
    tasks: dict[str, ArcTask], results: dict[str, Any], path: Path
) -> None:
    """Write a visual leave-one-task-out comparison for several ARC tasks."""

    sections = []
    for fold in results["folds"]:
        task = tasks[fold["holdout_task_id"]]
        trained = fold["trained_official_test"]
        untrained = fold["untrained_official_test"]
        corrupted = fold["corrupted_holdout_support"]
        demos = []
        for index, pair in enumerate(task.train, start=1):
            demos.append(
                '<div class="demo"><b>Demo '
                f'{index}</b><div class="grid-group">'
                f'<div><span>Input</span>{_grid(pair.input.tolist(), "input")}</div>'
                f'<div><span>Output</span>{_grid(pair.output.tolist(), "output")}</div>'
                "</div></div>"
            )
        tests = []
        for index, pair in enumerate(task.test):
            comparisons = (
                ("Test input", pair.input.tolist()),
                ("Expected", pair.output.tolist()),
                ("Untrained", untrained["predictions"][index]),
                ("Trained", trained["predictions"][index]),
                ("Corrupt support", corrupted["predictions"][index]),
            )
            tests.append(
                '<div class="comparison">'
                + "".join(
                    f'<div><span>{html.escape(label)}</span>{_grid(grid, label)}</div>'
                    for label, grid in comparisons
                )
                + "</div>"
            )
        status = "CORRECT" if trained["all_exact"] else "INCORRECT"
        sections.append(
            f'<section><h2>Held out: {html.escape(task.task_id)} — {status}</h2>'
            f'<p>Updater trained on {html.escape(", ".join(fold["training_task_ids"]))}. '
            f'Official cell accuracy: {trained["cell_accuracy"]:.1%}; '
            "test target excluded from training.</p>"
            f'<div class="demos">{"".join(demos)}</div>{"".join(tests)}</section>'
        )
    summary = html.escape(json.dumps(results["summary"], indent=2))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARC leave-one-task-out report</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#17202a}}main{{max-width:1250px;margin:auto}}
section{{background:white;padding:18px;margin:18px 0;border:1px solid #d8dee4;border-radius:9px}}h1,h2{{font-weight:650}}
.demos{{display:flex;gap:14px;flex-wrap:wrap}}.demo{{padding:10px;background:#f6f7f9;border-radius:6px}}
.grid-group,.comparison{{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;margin:12px 0}}
.grid-group>div>span,.comparison>div>span{{display:block;margin-bottom:5px;font-weight:600}}
.grid{{display:inline-block;border:2px solid #45525f;background:#45525f;line-height:0}}.grid-row{{display:flex}}
.cell{{width:25px;height:25px;margin:1px;display:inline-flex;align-items:center;justify-content:center;color:white;font-size:10px;line-height:1;text-shadow:0 1px 2px #000}}
pre{{background:#17202a;color:#eaf2f8;padding:12px;border-radius:8px;overflow:auto}}
</style></head><body><main><h1>Genuine ARC task-level transfer</h1>
<p>Each model learned its recurrent updater on three task IDs. The displayed fourth task was unseen during optimization; only its demonstrations were supplied at inference.</p>
<pre>{summary}</pre>{''.join(sections)}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def write_composition_report(results: dict[str, Any], path: Path) -> None:
    """Render held-out composition traces and predictions from both models."""

    sections = []
    for example in results["examples"]:
        supports = []
        for index, (source, target) in enumerate(
            zip(example["support_inputs"], example["support_targets"]), start=1
        ):
            supports.append(
                f'<div><b>Demo {index}</b><div class="grids">'
                f'<div><span>Input</span>{_grid(source, "support input")}</div>'
                f'<div><span>Output</span>{_grid(target, "support output")}</div>'
                "</div></div>"
            )
        comparisons = (
            ("Query", example["query_input"]),
            ("True intermediate", example["query_intermediate"]),
            ("Expected", example["query_target"]),
            ("I/O-only", example["models"]["io_only"]["prediction"]),
            (
                "Trace-supervised",
                example["models"]["trace_supervised"]["prediction"],
            ),
        )
        comparison_html = "".join(
            f'<div><span>{html.escape(label)}</span>{_grid(grid, label)}</div>'
            for label, grid in comparisons
        )
        io_trace = html.escape(example["models"]["io_only"]["predicted_trace"])
        supervised_trace = html.escape(
            example["models"]["trace_supervised"]["predicted_trace"]
        )
        sections.append(
            f'<section><h2>Held out: {html.escape(example["heldout_trace"])}</h2>'
            f'<div class="supports">{"".join(supports)}</div>'
            f'<div class="comparison">{comparison_html}</div>'
            f'<p><b>I/O-only trace:</b> {io_trace}<br>'
            f'<b>Trace-supervised trace:</b> {supervised_trace}</p></section>'
        )

    summary = {
        "untrained_heldout_exact": results["untrained_heldout"][
            "exact_grid_accuracy"
        ],
        "io_only_seen_exact": results["io_only"]["seen"]["exact_grid_accuracy"],
        "io_only_validation_exact": results["io_only"]["validation"][
            "exact_grid_accuracy"
        ],
        "io_only_heldout_exact": results["io_only"]["heldout"][
            "exact_grid_accuracy"
        ],
        "trace_seen_exact": results["trace_supervised"]["seen"][
            "exact_grid_accuracy"
        ],
        "trace_validation_exact": results["trace_supervised"]["validation"][
            "exact_grid_accuracy"
        ],
        "trace_heldout_exact": results["trace_supervised"]["heldout"][
            "exact_grid_accuracy"
        ],
        "trace_heldout_joint_operator": results["trace_supervised"]["heldout"][
            "joint_trace_accuracy"
        ],
    }
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARC unseen-composition experiment</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#17202a}}main{{max-width:1300px;margin:auto}}
section{{background:white;padding:18px;margin:18px 0;border:1px solid #d8dee4;border-radius:9px}}.supports{{display:flex;gap:18px;flex-wrap:wrap}}
.grids,.comparison{{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin:10px 0}}.grids>div>span,.comparison>div>span{{display:block;font-weight:600;margin-bottom:5px}}
.grid{{display:inline-block;border:2px solid #45525f;background:#45525f;line-height:0}}.grid-row{{display:flex}}
.cell{{width:24px;height:24px;margin:1px;display:inline-flex;align-items:center;justify-content:center;color:white;font-size:10px;line-height:1;text-shadow:0 1px 2px #000}}
pre{{background:#17202a;color:#eaf2f8;padding:12px;border-radius:8px;overflow:auto}}
</style></head><body><main><h1>Learned operator encoder: unseen compositions</h1>
<p>Both models receive raw demonstration pairs. Selected spatial-plus-color pairings are absent from training; only one model receives operator-token and intermediate-grid traces.</p>
<pre>{html.escape(json.dumps(summary, indent=2))}</pre>{''.join(sections)}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
