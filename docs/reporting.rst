Reporting
=========

Flowcept can generate summarized reports from provenance records.

Current report implementation:

- ``report_type="provenance_card"``
- ``report_type="provenance_report"``
- ``provenance_card`` uses ``format="markdown"``
- ``provenance_report`` uses ``format="pdf"`` (executive PDF with plots)


API
---

Use:

.. code-block:: python

   from flowcept import Flowcept

   Flowcept.generate_report(
       report_type="provenance_card",
       format="markdown",
       output_path="PROVENANCE_CARD.md",
       records=my_records,  # or input_jsonl_path=..., or workflow_id/campaign_id
   )

   # PDF report (optional dependency: flowcept[report_pdf])
   Flowcept.generate_report(
       report_type="provenance_report",
       format="pdf",
       output_path="PROVENANCE_REPORT.pdf",
       workflow_id="my_workflow_id",
   )


Input Modes
-----------

Exactly one input mode must be provided:

- ``input_jsonl_path``: read from a Flowcept JSONL buffer file.
- ``records``: list of dictionaries already loaded in memory.
- ``workflow_id`` or ``campaign_id``: query workflow, task, and object documents from DB.


Aggregation
-----------

The provenance card is summarized, not raw-dump oriented.

- Grouping key: ``activity_id``.
- Per-group summary includes:
  - number of task records aggregated (``n_tasks``)
  - status counts
  - timing aggregates (average and maximum elapsed)

This aggregation method is also written in the generated card under
``Aggregation Method``.


PDF Report Notes
----------------

The PDF renderer keeps the provenance card markdown content and adds executive plots:

- Top slowest activities (bar chart)
- Top fastest activities (bar chart)
- Most resource-demanding activities by total IO bytes (bar chart)

Install optional dependencies:

.. code-block:: shell

   pip install flowcept[report_pdf]


Object Metadata Summary
-----------------------

When objects are present, the card includes metadata-only summaries:

- counts by type
- counts by storage mode (``in_object`` vs ``gridfs``)
- linkage counts (task/workflow-linked)
- maximum observed object version

Blob payload bytes are excluded from report rendering.


Example Generated Provenance Card
---------------------------------

Below is an example of a generated workflow provenance card (with user anonymized).

.. note::

   This is an illustrative snapshot. Exact values, available telemetry fields, and
   some optional sections depend on your runtime environment, enabled telemetry,
   and active storage/backend configuration.

.. code-block:: markdown

   # Workflow Provenance Card: Perceptron Train

   ## Summary
   - **Workflow Name:** `Perceptron Train`
   - **Workflow ID:** `c5096d4a-5978-43c5-86a6-9bd8f5469b56`
   - **Campaign ID:** `e7cdfd1a-b026-4e34-8ce2-b9d4b09e9dac`
   - **Execution Start (UTC):** `2026-02-18 00:08:35`
   - **Execution End (UTC):** `2026-02-18 00:08:36`
   - **Total Elapsed (s):** `1.208`
   - **User:** `anonymous`
   - **System Name:** `Darwin`
   - **Environment ID:** `laptop`
   - **Code Repository:** `branch=dev, short_sha=ad8f4df, dirty=dirty`
   - **Git Remote:** `git@github.com:ORNL/flowcept.git`
   - **Workflow args:**
     <br> `python_random_seeded`: `True`
     <br> `seed`: `42`
     <br> `torch_cuda_manual_seeded`: `False`
     <br> `torch_cudnn_benchmark`: `False`
     <br> `torch_cudnn_deterministic`: `True`
     <br> `torch_deterministic_algorithms`: `True`
     <br> `torch_manual_seeded`: `True`

   ## Workflow-level Summary
   - **Total Activities:** `2`
   - **Status Counts:** `{'FINISHED': 2}`
   - **Total Elapsed Workflow Time (s):** `1.208`
     - `train_and_validate`: `1.180 s`
     - `get_dataset`: `0.027 s`
   - **Resource Totals:**
     - `Memory Used`: `47.34 MB`
     - `Average CPU (%)`: `61.1%`
     - **IO:**
       - `Read`: `51.83 MB`
       - `Write`: `29.78 MB`
       - `Read Ops`: `2,031`
       - `Write Ops`: `518`
   - **Key Observations:**
     - Slowest Activity: `train_and_validate` at `1.180 s`
     - Largest IO Activity: `train_and_validate` with Read `45.37 MB` and Write `29.78 MB`

   ## Workflow Structure

   ```text
   input data
           │
           ▼
    get_dataset
           │
    train_and_validate
           ▼
   output data
   ```

   ## Timing Report
   Rows are sorted by **Started At** (ascending).

   | Activity | Status | Started At | Ended At | Elapsed (s) |
   | --- | --- | --- | --- | --- |
   | get_dataset | FINISHED | 2026-02-18 00:08:35 | 2026-02-18 00:08:35 | 0.027 |
   | train_and_validate | FINISHED | 2026-02-18 00:08:35 | 2026-02-18 00:08:36 | 1.180 |

   ## Per Activity Details
   - **get_dataset** (`n=1`)
     - Used (aggregated):
       - `n_samples`: presence=100.0%; type=numeric; min=120.000; p50=120.000; p95=120.000; max=120.000
       - `split_ratio`: presence=100.0%; type=numeric; min=0.800; p50=0.800; p95=0.800; max=0.800
   - **train_and_validate** (`n=1`)
     - Used (aggregated):
       - `checkpoint_check`: presence=100.0%; type=numeric; min=2.000; p50=2.000; p95=2.000; max=2.000
       - `epochs`: presence=100.0%; type=numeric; min=6.000; p50=6.000; p95=6.000; max=6.000
       - `n_input_neurons`: presence=100.0%; type=numeric; min=2.000; p50=2.000; p95=2.000; max=2.000

   ## Workflow-level Resource Usage
   | Metric | Value |
   | --- | --- |
   | Telemetry Samples (task start/end pairs) | 2 |
   | CPU User Time Delta | 4.240 |
   | CPU System Time Delta | 3.100 |
   | Average CPU (%) Delta | 61.1% |
   | Average CPU Frequency | 3,228 |
   | Memory Used Delta | 47.34 MB |
   | Average Memory (%) | 84.4% |
   | Swap Used Delta | 21.12 MB |
   | Average Swap (%) | 95.7% |
   | Disk Read Time Delta (ms) | 479.000 |
   | Disk Write Time Delta (ms) | 15.000 |

   ## Per-activity Resource Usage
   | Activity | Elapsed (s) | CPU User (s) | CPU System (s) | CPU (%) | Memory Delta | Read | Write | Read Ops | Write Ops |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | get_dataset | 0.027 | unknown | unknown | - | - | 6.46 MB | - | 86 | - |
   | train_and_validate | 1.180 | 4.240 | 3.100 | 61.1% | 47.34 MB | 45.37 MB | 29.78 MB | 1,945 | 518 |

   ## Object Artifacts Summary
   | Metric | Value |
   | --- | --- |
   | Total Objects | 3 |
   | By Type | {'dataset': 1, 'ml_model': 2} |
   | By Storage | {'in_object': 2, 'gridfs': 1} |
   | Task-linked Objects | 3 |
   | Workflow-linked Objects | 3 |
   | Max Version | 2 |
   | Total Size | 6.79 KB |
   | Average Size | 2.26 KB |
   | Max Size | 4.10 KB |

   ## Aggregation Method
   - Grouping key: `activity_id`.
   - Each grouped row may aggregate multiple task records (`n_tasks`).
   - Aggregated metrics currently include count/status/timing.
