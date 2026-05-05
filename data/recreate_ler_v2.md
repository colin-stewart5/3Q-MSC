# Recreating `ler_v2.png`

This repo does not include the original figure-building script, but it does include the aggregated dataset that the figure was built from: `MSC_foldedH/data/all_paper_data_v2.csv`.

The script below reconstructs the checked-in plot from that CSV without changing any existing simulation code:

- `MSC_foldedH/data/recreate_ler_v2.py`

It reproduces the figure by:

1. Parsing the older 8-column `sinter` CSV format stored in `all_paper_data_v2.csv`.
2. Rebuilding each curve from the `custom_counts` gap histogram.
3. Plotting the same comparison set used in the checked-in image:
   - `This work` at fault distance 3 and 5
   - `Exact simulation`
   - `GJS24`
   - `CCLP25`
4. Encoding fault distance by marker shape:
   - triangle = `f=3`
   - pentagon = `f=5`

## Commands

From the repo root:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC
python3 MSC_foldedH/data/recreate_ler_v2.py \
  --csv MSC_foldedH/data/all_paper_data_v2.csv \
  --output MSC_foldedH/data/ler_v2.recreated.png
```

If you want to overwrite the checked-in file directly:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC
python3 MSC_foldedH/data/recreate_ler_v2.py \
  --csv MSC_foldedH/data/all_paper_data_v2.csv \
  --output MSC_foldedH/data/ler_v2.png
```

## Environment

The script only needs `matplotlib` plus the Python standard library.

If your active Python does not already have `matplotlib`, install it into a virtual environment first:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC
python3 -m venv .venv-ler
source .venv-ler/bin/activate
python -m pip install matplotlib
python MSC_foldedH/data/recreate_ler_v2.py \
  --csv MSC_foldedH/data/all_paper_data_v2.csv \
  --output MSC_foldedH/data/ler_v2.recreated.png
```

## Notes

- This recreates the plot from the checked-in aggregate CSV, not by rerunning the full Monte Carlo sampling campaign.
- That is the reliable path available in this repo today; the original raw figure script is not present.

## Patch Status

I patched the repo code in three places while keeping the original workflow intact:

- `MSC_foldedH/src/full_clifford_sim/gap_sampler.py`
  - cast numpy scalar counts into plain Python `int` / `float` before constructing `sinter.AnonTaskStats`
  - this fixes the crash that blocked `handoff_run_3err.py`
- `MSC_foldedH/src/full_clifford_sim/_noise.py`
  - allow the specific “gate then measure/reset on the same qubit in one moment” pattern used by the GHZ-5 flag circuit
  - this fixes the d=5 `|Y>` crash without inserting an extra `TICK`
- `MSC_foldedH/src/handoff_run_3err.py`
  - replaced the hardcoded two-argument parsing with a real CLI
  - added optional controls for `--dfinal`, `--tag`, `--fault-distance-tag`, and other run parameters
  - preserved the original positional behavior: `handoff_run_3err.py <p_milli> <tasknum>`

## Rebuilding The Underlying CSV

`all_paper_data_v2.csv` is not a pure “one script generated everything” file.

- Some rows are local outputs from this repo.
- Some rows are comparison data from prior work that are already checked in.
- For the figure used by `recreate_ler_v2.py`, the locally-generated rows are the `This work` and `Exact simulation` entries.
- The `GJS24` and `CCLP25` rows are comparison rows, not something this repo currently regenerates from a local driver.

The practical way to proceed is:

1. Get the cheaper `|Y>` path working first.
2. Verify you can emit headerless `TaskStats.to_csv_line()` rows.
3. Only then scale shot counts up and move on to the `|H_XY>` exact-simulation / handoff path.

## Tested Environment For Data Generation

The plotting script only needed `matplotlib`, but the data-generation path needed more:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC
./.venv/bin/pip install numpy matplotlib stim sinter pymatching cirq stimcirq
```

For the commands below, I used:

```bash
MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=MSC_foldedH/src
./.venv/bin/python3.12
```

`PYTHONPATH=MSC_foldedH/src` is required so the local `full_clifford_sim` package imports correctly.

## Cheapest Working Path: `|Y>` Data First

The closest existing repo driver is `MSC_foldedH/src/full_clifford_sim/main_fullcirc_run.py`, but it is not a general CLI. It is hardcoded to one task and very large shot counts:

- `prep="hookinj"`
- `glen=3`
- `l=3`
- `df=13`
- `p=0.001`
- `max_shots=80_000_000`

For a real probe without editing repo code, I had better results using a scratch driver file that imports the same functions.

Important detail:

- On this macOS / Python 3.12+ setup, `sinter.collect(...)` should be run from a real `.py` file.
- Running it from `python - <<'PY'` failed because the worker process respawned from `<stdin>`.

### Minimal d=3 `|Y>` Probe

Create a temporary driver:

```bash
cat > /tmp/run_y_probe.py <<'PY'
import sinter
from full_clifford_sim.main_complied_fxns import full_circuit
from full_clifford_sim.gap_sampler import sinter_samplers

task = sinter.Task(
    circuit=full_circuit(
        0.001,
        dfinal=13,
        ghz_size=3,
        latter_rounds=0,
        prep="unitstab",
        ps_on_d3=1,
    ),
    json_metadata={
        "p": 0.001,
        "b": "Y",
        "noise": "uniform",
        "d2": 13,
        "c": "e2eY-unitstab-g3-3ps1_mid7x3-ue13x0-uniform",
        "ghz_size": 3,
        "latter_rounds": 0,
        "middle_rounds": 3,
        "prep": "unitstab",
        "ps3": 1,
        "f": 3,
    },
)

res = sinter.collect(
    tasks=[task],
    num_workers=1,
    decoders=["pymatching-gap"],
    custom_decoders=sinter_samplers(),
    max_shots=1000,
    max_errors=100,
    print_progress=False,
)

for stat in res:
    print(stat.to_csv_line())
PY
```

Run it:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=MSC_foldedH/src \
./.venv/bin/python3.12 /tmp/run_y_probe.py
```

That path worked for me and emitted a valid headerless CSV line in the same 8-column layout used by `all_paper_data_v2.csv`.

### Scaling The `|Y>` Path

Once the single-task probe works, scale by expanding:

- `p`
- `prep`
- `ghz_size`
- `latter_rounds`
- `dfinal`
- metadata fields like `c`, `f`, `ps3`, and `middle_rounds`

If you want to build a file in the same style as `all_paper_data_v2.csv`, append `stat.to_csv_line()` outputs yourself. Do not rely on `save_resume_filepath` if you want the exact same layout, because current `sinter.collect(..., save_resume_filepath=...)` writes a header row while `all_paper_data_v2.csv` is headerless.

The notebook `MSC_foldedH/src/testing_circ_generation.ipynb` already contains a smaller `10_000`-shot `sinter.collect(...)` example for the `|Y>` path, which is useful for sanity-checking the workflow before scaling up.

## What Happened When I Tried d=5 `|Y>`

Before patching, the obvious d=5 `|Y>` path failed with:

```text
ValueError: Qubits were operated on multiple times without a TICK in between
```

coming from `full_clifford_sim/_noise.py`.

After the noise-model patch, the same path runs.

I tested the obvious metadata-matching call for the local d=5 row:

- `dfinal=13`
- `ghz_size=5`
- `latter_rounds=4`
- `prep="unitstab"`
- `p=0.001`

It now produces valid `TaskStats` output. A small probe command is:

```bash
cat > /tmp/run_y_fd5_probe.py <<'PY'
import sinter
from full_clifford_sim.main_complied_fxns import full_circuit
from full_clifford_sim.gap_sampler import sinter_samplers

task = sinter.Task(
    circuit=full_circuit(
        0.001,
        dfinal=13,
        ghz_size=5,
        latter_rounds=4,
        prep="unitstab",
    ),
    json_metadata={
        "p": 0.001,
        "b": "Y",
        "noise": "uniform",
        "d2": 13,
        "c": "e2e-cls-Y-unitstab-hybrid-fd5-df13",
        "ghz_size": 5,
        "latter_rounds": 4,
        "prep": "unitstab",
        "f": 5,
        "ourbase": 1,
    },
)

res = sinter.collect(
    tasks=[task],
    num_workers=1,
    decoders=["pymatching-gap"],
    custom_decoders=sinter_samplers(),
    max_shots=2000,
    max_errors=100,
    print_progress=False,
)

for stat in res:
    print(stat.to_csv_line())
PY
```

Run it with:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=MSC_foldedH/src \
./.venv/bin/python3.12 /tmp/run_y_fd5_probe.py
```

### Comparison Against The Plot Data

With the patched code, the path runs but does **not** numerically reproduce the checked-in published d=5 row from this repo snapshot.

My `2000`-shot probe produced:

- attempts at zero gap threshold: about `1.90`
- logical error rate at zero gap threshold: about `3.8e-3`

The checked-in figure row `e2e-cls-Y-unitstab-hybrid-fd5-df13` has:

- attempts at zero gap threshold: about `3.21`
- logical error rate at zero gap threshold: about `9.5e-4`

That mismatch is too large to attribute to sampling noise.

My conclusion is that the crash is fixed, but the exact published fd=5 hybrid protocol is not fully reconstructible from the code paths exposed in this repo. In particular, the paper’s fd=5 protocol adds an extra grow-to-`Rot(5)` / check-on-`Reg(5)` structure, and the checked-in Python assembly code does not currently expose a dedicated builder for that full published workflow.

So:

- the d=5 `|Y>` path now runs,
- but it should be treated as a repaired runnable path, not as a confirmed reproduction of the published fd=5 curve.

## Experience With `handoff_run_3err.py`

`MSC_foldedH/src/handoff_run_3err.py` is the natural place to look for the `|H_XY>` / exact-simulation rows.

### What The Script Does

It is not a generic runner. It hardcodes several choices:

- `basis = 'HXY'`
- `prep = 'unitstab'`
- `d2 = 7`
- `ghz_size = 3`
- `num_sverr_shots = 10`
- `num_shots_per_sverr = 20`

Its CLI is:

```bash
./.venv/bin/python3.12 MSC_foldedH/src/handoff_run_3err.py <p_milli> <tasknum>
```

where:

- `<p_milli>` is interpreted as `0.001 * int(sys.argv[1])`
- `<tasknum>` is used as the simulator seed / HPC shard id

So:

```bash
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=MSC_foldedH/src \
./.venv/bin/python3.12 MSC_foldedH/src/handoff_run_3err.py 1 0
```

means `p = 0.001` and `tasknum = 0`.

### What Happened In Practice

Before patching, my first attempt failed immediately because the default environment here did not have `cirq` or `stimcirq`.

After installing those into `.venv`, the script got much farther:

- it built the reference circuit,
- converted the truncated stim circuit into Cirq,
- ran the statevector stage,
- wrote intermediate JSON into a dated directory like `realcirqhandoffmid_p1_outsMMDD/`,
- then entered the `sinter.collect(...)` post-processing stage.

It then crashed with:

```text
AssertionError
...
assert isinstance(self.errors, int)
```

inside `sinter.AnonTaskStats`, triggered from `MSC_foldedH/src/full_clifford_sim/gap_sampler.py`.

After patching `gap_sampler.py`, the script runs cleanly.

### Reproducing The Plot-Row Shape

The patched CLI can now target the checked-in plot-row metadata directly:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=MSC_foldedH/src \
./.venv/bin/python3.12 MSC_foldedH/src/handoff_run_3err.py \
  1 0 \
  --dfinal 13 \
  --tag 1002 \
  --fault-distance-tag 3
```

This produces metadata of the form:

```json
{"b":"HXY","c":"Tcirq-handoff-1002","d2":13,"f":3,"ghz_size":3,"latter_rounds":3,"noise":"uniform","p":0.001}
```

matching the row family used by the plot.

### Comparison Against The Plot Data

I validated the patched handoff runner against the checked-in `Tcirq-handoff-1002` row in three ways.

Single shard:

- one `tasknum` run writes a valid 8-column CSV row and the expected dated intermediate/output files
- this is the basic “it runs” verification

Small aggregate:

- I ran `tasknum=0..9` with `--dfinal 13 --tag 1002 --fault-distance-tag 3`
- aggregated result: `shots=2000`, `errors=4`, `discards=594`
- aggregate zero-threshold attempts: about `1.42`
- aggregate zero-threshold logical error rate: about `2.84e-3`

Contiguous larger aggregate:

- I then ran a contiguous `tasknum=0..399` sample, all with `--dfinal 13 --tag 1002 --fault-distance-tag 3`
- this produced `400` shard files in `realcirqhandoffoutputs_p1_1002/`
- aggregate result: `shots=80000`, `errors=209`, `discards=26459`
- aggregate zero-threshold attempts: about `1.494`
- aggregate zero-threshold logical error rate: about `3.90e-3`
- approximate 95% interval on the aggregate zero-threshold logical error rate: about `[3.38e-3, 4.43e-3]`
- approximate 95% interval on the aggregate zero-threshold attempts: about `[1.491, 1.508]`

The checked-in plot row `Tcirq-handoff-1002` has:

- zero-threshold attempts: about `1.60`
- zero-threshold logical error rate: about `4.09e-3`

Interpretation:

- the patched handoff path is now clearly producing the correct row family and the correct order of magnitude for the logical error curve
- the larger aggregate logical error rate is statistically consistent with the checked-in row
- the acceptance / discard rate is still noticeably different, so this is not yet an exact reproduction of the historical handoff dataset
- the checked-in row also uses a historical non-hash identifier string (`octrealcirqhandoff3q-1002decodinglog`) in column 6, while the current script writes a hash-based `strong_id`; that packaging difference does not affect the plotted metadata or counts, but it means the emitted row is not byte-for-byte identical to the checked-in CSV entry

Determinism check:

- rerunning the same `tasknum` does not reproduce the same shard exactly
- for example, two separate reruns of `tasknum=500` produced `discards=33` and `discards=37`, with different `custom_counts`
- that means exact historical row recreation is only meaningful at the level of the aggregate distribution, not at the level of a single shard file

To reproduce the same aggregate I used for the comparison above:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC
seq 0 399 | xargs -n1 -P8 -I{} env \
  MPLCONFIGDIR=/tmp/mpl \
  PYTHONPATH=MSC_foldedH/src \
  ./.venv/bin/python3.12 MSC_foldedH/src/handoff_run_3err.py \
  1 {} --dfinal 13 --tag 1002 --fault-distance-tag 3
```

and then aggregated the emitted CSV-row files under `realcirqhandoffoutputs_p1_1002/`.

### Why This Matters For Reproducing `all_paper_data_v2.csv`

- The patched script no longer forces the metadata label to the current date; `--tag` lets you recreate historical row names such as `Tcirq-handoff-1002`.
- The output directories also follow the same tag.
- Each `handoff_run_3err.py` invocation is still only one shard. Reproducing the large checked-in handoff rows requires many `tasknum` runs followed by aggregation of their `TaskStats.to_csv_line()` outputs.

## Validation Summary

After patching:

- d=3 `|Y>` path runs and matches the checked-in plot row closely at small-shot probe level
  - my `5000`-shot probe gave about `1.51` attempts and `2.7e-3` LER at zero threshold
  - the checked-in row is about `1.49` attempts and `3.6e-3` LER
- `handoff_run_3err.py` now runs and can target the `Tcirq-handoff-1002` row family directly
  - a contiguous `tasknum=0..399` run gave about `1.494` attempts and `3.90e-3` LER
  - the checked-in row is about `1.598` attempts and `4.09e-3` LER
  - the logical error rate now matches within sampling error, but the discard / acceptance rate still does not
- d=5 `|Y>` no longer crashes
  - but the obvious runnable path does not reproduce the published fd=5 row from `all_paper_data_v2.csv`
  - that appears to be a missing protocol-assembly gap in the repo, not just a runtime bug

## Handoff Discrepancy Analysis

I specifically investigated why the patched `d2=13` handoff workflow still misses the checked-in `Tcirq-handoff-1002` discard rate.

### Control Case: The Original `d2=7` Path Is Correct

The repo also contains a historical `d2=7` handoff row:

- `Tcirq-handoff-0212`

I reran that path with the current patched script over `tasknum=0..99`:

- sample result: `shots=20000`, `errors=57`, `discards=6536`
- zero-threshold attempts: about `1.485`
- zero-threshold logical error rate: about `4.23e-3`

The checked-in row has:

- zero-threshold attempts: about `1.488`
- zero-threshold logical error rate: about `4.05e-3`

That match is close enough that I do not think the general handoff machinery is wrong. The current codebase reproduces the original `d2=7` row family.

### What Is Actually Mismatching At `d2=13`

For the `d2=13` `Tcirq-handoff-1002` comparison:

- the accepted-shot threshold spectrum matches the checked-in row very closely once you condition on acceptance
- the total-variation distance between the accepted-shot bucket masses `(Ck + Ek)` and the checked-in row is only about `8e-3`
- the logical error rate also lands in the correct range
- the main remaining mismatch is simply that too few shots are being discarded

In the contiguous `tasknum=0..399` run:

- total shots: `80000`
- total discards: `26459`
- accepted shots: `53541`
- zero-threshold attempts: about `1.494`

The checked-in row has:

- total shots: `800000000`
- total discards: `299464869`
- accepted shots: `500535131`
- zero-threshold attempts: about `1.598`

### Stage Split Of The Current `d2=13` Run

Using the intermediate `realcirqhandoffmid_p1_outs1002/` files, I decomposed the current `d2=13` discards:

- statevector-stage postselection rejects: `757 / 4000` exact-sim shots, about `18.9%`
- downstream Clifford / gap-sampler discards: `11319 / 64860` launched shots, about `17.5%`

So the current pipeline is internally consistent. To reach the checked-in total acceptance while keeping the downstream behavior fixed, the exact-simulation postselection stage would need to reject noticeably more often than it currently does.

### Variants I Checked

- `prep=hookinj` is not the answer; at `d2=13` it produces a completely different and much worse acceptance profile
- increasing `latter_rounds` raises the discard rate, but it also pulls the accepted-shot spectrum away from the checked-in `Tcirq-handoff-1002` row, and the historical metadata explicitly says `latter_rounds=3`
- the checked-in `Tcirq-handoff-1002` row also uses a non-hash identifier string (`octrealcirqhandoff3q-1002decodinglog`) instead of the direct `TaskStats` hash emitted by the current script, which is another sign that the historical `d2=13` row likely came from a slightly different external aggregation / runner path

### Current Best Explanation

The evidence points to this:

- the checked-in `d2=13` handoff row was probably not produced by the exact checked-in `handoff_run_3err.py` workflow as it stands
- the accepted-shot physics is close, so the missing piece is most likely an extra rejection / postselection effect in the exact-simulation stage or a slightly different historical runner configuration
- the repo snapshot is good enough to reproduce the original `d2=7` handoff workflow, but not enough to reproduce the historical `d2=13` handoff row exactly

## Session Update: March 26, 2026

I reran the reconstruction workflow without changing repo code.

Important constraint:

- `handoff_run_3err_ccz.py` was intentionally ignored in this pass
- the goal here was only to recreate the original `ler_v2` / `all_paper_data_v2.csv` local rows as far as this snapshot allows

### Working Environment Right Now

The usable environment in this workspace is:

```bash
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=MSC_foldedH/src \
./.venv312/bin/python3
```

As of this session, `.venv312` has:

- `numpy`
- `matplotlib`
- `stim`
- `sinter`
- `pymatching`
- `cirq`
- `stimcirq`

The repo-root `.venv` was not useful for this workflow.

### Fresh No-Code Reproduction Results

I regenerated a small set of representative local rows and compared them directly against the matching rows in `MSC_foldedH/data/all_paper_data_v2.csv`.

#### Direct `|Y>` Rows At `p=0.001`

These were recreated from `full_circuit(...)` plus `sinter.collect(...)` using a temporary driver file, not by editing repo code.

`e2eY-unitstab-g3-3ps1_mid7x3-ue13x0-uniform`

- fresh run: `shots=5000`, `errors=13`, `discards=1674`
- fresh zero-threshold attempts: about `1.503`
- fresh zero-threshold logical error rate: about `3.91e-3`
- checked-in row: attempts about `1.488`, logical error rate about `3.56e-3`

`e2eY-hookinj-g3-3ps1_mid7x3-ue13x0-uniform`

- fresh run: `shots=5000`, `errors=11`, `discards=1886`
- fresh zero-threshold attempts: about `1.606`
- fresh zero-threshold logical error rate: about `3.53e-3`
- checked-in row: attempts about `1.613`, logical error rate about `3.56e-3`

`e2eY-optunit-g3-3ps1_mid7x3-ue13x0-uniform`

- fresh run: `shots=5000`, `errors=12`, `discards=1609`
- fresh zero-threshold attempts: about `1.474`
- fresh zero-threshold logical error rate: about `3.54e-3`
- checked-in row: attempts about `1.455`, logical error rate about `3.56e-3`

Interpretation:

- the direct `d=3` `|Y>` rows are still reproducible from this repo
- `hookinj` and `optunit` matched very closely
- `unitstab` was also in-family, though the small-shot rerun landed a bit high on LER

#### Published `fd=5` Hybrid `|Y>` Row

`e2e-cls-Y-unitstab-hybrid-fd5-df13`

- fresh run: `shots=2000`, `errors=1`, `discards=944`
- fresh zero-threshold attempts: about `1.894`
- fresh zero-threshold logical error rate: about `9.47e-4`
- checked-in row: attempts about `3.212`, logical error rate about `9.51e-4`

Interpretation:

- the zero-threshold logical error rate still lands in the right place
- the acceptance / discard behavior is still badly off
- this confirms the earlier conclusion: the code path runs, but the historical published `fd=5` hybrid row is still not fully reconstructible from this repo snapshot

#### Handoff `|H_XY>` Rows

I reran the existing handoff driver with fresh tags and aggregated shard outputs by summing:

- `shots`
- `errors`
- `discards`
- `custom_counts`

Each shard is still one invocation of:

```bash
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=MSC_foldedH/src \
./.venv312/bin/python3 MSC_foldedH/src/handoff_run_3err.py ...
```

##### Fresh `d2=13` Reproduction Of The `1002` Row Family

I ran:

- `tasknum=0..99`
- `p=0.001`
- `--dfinal 13`
- `--tag repro0326b`
- `--fault-distance-tag 3`

Aggregate result:

- `shots=20000`
- `errors=53`
- `discards=6560`
- zero-threshold attempts: about `1.488`
- zero-threshold logical error rate: about `3.94e-3`

Checked-in `Tcirq-handoff-1002` row:

- zero-threshold attempts: about `1.598`
- zero-threshold logical error rate: about `4.09e-3`

Interpretation:

- the logical error rate is again in the correct range
- the acceptance mismatch remains
- this fresh rerun agrees with the earlier conclusion that the current repo reproduces the row family, but not the historical `d2=13` acceptance exactly

##### Fresh `d2=7` Control Reproduction Of The `0212` Row Family

I also reran the old `d2=7` control path:

- `tasknum=0..99`
- `p=0.001`
- default `dfinal=7`
- `--tag repro0212c`

Aggregate result:

- `shots=20000`
- `errors=65`
- `discards=6534`
- zero-threshold attempts: about `1.485`
- zero-threshold logical error rate: about `4.83e-3`

Checked-in `Tcirq-handoff-0212` row:

- zero-threshold attempts: about `1.488`
- zero-threshold logical error rate: about `4.05e-3`

Interpretation:

- the acceptance matches very closely
- the fresh LER landed a bit high at this shot count
- this still supports the main structural point: the original `d2=7` handoff workflow is fundamentally reproducible in this repo

### What Fresh Plot Reconstruction Showed

I also built a temporary mixed CSV consisting of:

- freshly regenerated local rows for:
  - `e2eY-unitstab-g3-3ps1_mid7x3-ue13x0-uniform`
  - `e2e-cls-Y-unitstab-hybrid-fd5-df13`
  - `Tcirq-handoff-1002` (using the normalized fresh aggregate metadata label)
- checked-in comparison rows for:
  - `GJS24`
  - `CCLP25`

Running `MSC_foldedH/data/recreate_ler_v2.py` on that mixed CSV confirmed:

- the fresh local rows sit in the right zero-threshold regime
- with only `2000` to `20000` fresh shots, the recreated local curves do not extend nearly as far into the ultra-low-LER tail as the published plot
- in particular, the published checked-in rows reach much larger max-attempt ranges because they were generated with vastly larger shot counts

So:

- this repo can still reproduce the local row families qualitatively and numerically near zero threshold
- but not yet the full published plot tails from small reruns

### Fresh Workspace Artifacts

The most useful fresh directories created in this session are:

- `realcirqhandoffmid_p1_outsrepro0326b`
- `realcirqhandoffoutputs_p1_repro0326b`
- `realcirqhandoffmid_p1_outsrepro0212c`
- `realcirqhandoffoutputs_p1_repro0212c`

The `repro0326b` directories are the fresh `d2=13` handoff rerun.

The `repro0212c` directories are the fresh `d2=7` control rerun.

### Current Bottom Line

For future sessions, the current state should be treated as:

- direct `d=3` `|Y>` rows are reproducible from the repo
- the historical `d2=7` handoff path is reproducible enough to serve as a control case
- the historical `d2=13` handoff row family is reproducible, but still has the known acceptance mismatch
- the published `fd=5` hybrid `|Y>` row still is not fully reconstructible from this repo snapshot
- no further repo code edits were needed to reach this point; the main blocker is protocol / historical-run mismatch, not another obvious runtime bug

## Session Update: April 16, 2026

I inspected `MSC_foldedH/src/handoff_run_3err_ccz.py` directly and also rebuilt both checked-in plot families into temporary files outside the repo.

### What `handoff_run_3err_ccz.py` Actually Changes

`handoff_run_3err_ccz.py` is structurally the same two-stage workflow as `handoff_run_3err.py`:

1. build a truncated Cirq exact-simulation circuit,
2. record logical Bloch vectors keyed by destabilizer / error location,
3. resample the corresponding Stim circuits with `sinter`,
4. fold the sampled gap counts back into a final `TaskStats` row.

The main additions are:

- replace the hardcoded ideal `cirq.CCZ(...)` calls with a configurable `ccz_factory(...)`
- add metadata tags:
  - `ccz_label`
  - `use_custom_ccz`
  - `custom_ccz_mode`
- write outputs into CCZ-labeled directories such as:
  - `realcirqhandoffoutputs_rydberg_ccz_pulse2_p1_<tag>/`

### Important Physics / Runtime Detail

With the matrix currently hardcoded in `handoff_run_3err_ccz.py`:

- `CUSTOM_CCZ_MODE = "auto"`
- `CUSTOM_CCZ_LABEL = "rydberg_ccz_pulse2"`
- `max(abs(U^\dagger U - I))` is about `6.42e-5`
- the singular values are about `[0.9999679, 1.0000004]`

That means:

- the matrix is **not** exactly unitary at the strict `1e-8` check,
- but it **does** pass the script's looser `CUSTOM_CCZ_NEAR_UNITARY_ATOL = 1e-4` check,
- so `build_ccz_factory()` projects it onto the nearest exact unitary with SVD,
- and the script stays on `cirq.Simulator`, **not** `cirq.DensityMatrixSimulator`.

So the currently configured "Rydberg CCZ" path is behaving as a coherent near-unitary pulse projected back to a unitary gate, not as a leakage / contractive Kraus-channel simulation.

### Recreating `ccz_vs_f3.png`

There is **no checked-in generator script** for `MSC_foldedH/data/ccz_vs_f3.png`.

The checked-in image can still be reconstructed from the data already present in this workspace by combining:

- checked-in rows from `MSC_foldedH/data/all_paper_data_v2.csv`
  - `|Y>`:
    - `c = "e2eY-unitstab-g3-3ps1_mid7x3-ue13x0-uniform"`
    - `p = 0.001`
    - `ourbase = 1`
  - `|H_XY>` ideal CCZ:
    - `c = "Tcirq-handoff-1002"`
    - `p = 0.001`
    - `f = 3`
  - reference curves:
    - `GJS24`, `p = 0.001`, `f = 3`
    - `CCLP25`, `p = 0.001`, `f = 3`
- plus the aggregated custom-CCZ handoff shards under:
  - `../realcirqhandoffoutputs_rydberg_ccz_pulse2_p1_cczovernight0327d/`

That external directory is important:

- it lives in the parent `3Q-MSC` workspace, not inside `MSC_foldedH/`
- so the repo by itself does **not** contain all the raw data needed to rebuild `ccz_vs_f3.png`

### Custom-CCZ Aggregate Used For The Plot

The strongest available custom-CCZ family in this workspace is:

- `../realcirqhandoffoutputs_rydberg_ccz_pulse2_p1_cczovernight0327d/`

It contains:

- `5500` shard files
- `220000000` total shots
- `71895914` discards
- `607692` logical errors

At zero gap threshold that aggregate gives approximately:

- expected attempts: `1.48544`
- logical error rate: `4.103e-3`

For comparison, the checked-in ideal exact-simulation row `Tcirq-handoff-1002` has:

- expected attempts: `1.59829`
- logical error rate: `4.091e-3`

So the custom-CCZ curve is very close in zero-threshold logical error rate, but with noticeably better acceptance.

### Recreating The Plots Without Writing Into The Repo

Working commands that succeeded in this workspace:

- `ler_v2`:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC/MSC_foldedH
../.venv312/bin/python3 data/recreate_ler_v2.py \
  --csv data/all_paper_data_v2.csv \
  --output /tmp/ler_v2_test.png
```

- custom-CCZ handoff smoke test:

```bash
cd /tmp
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=/Users/colinstewart/Desktop/WORK/3Q-MSC/MSC_foldedH/src \
/Users/colinstewart/Desktop/WORK/3Q-MSC/.venv312/bin/python3 \
  /Users/colinstewart/Desktop/WORK/3Q-MSC/MSC_foldedH/src/handoff_run_3err_ccz.py \
  1 0 \
  --dfinal 13 \
  --tag codexcczsmoke0416 \
  --fault-distance-tag 3 \
  --num-sverr-shots 1 \
  --num-shots-per-sverr 1
```

That smoke test completed and emitted a valid one-shot `TaskStats` row with metadata:

```json
{"b":"HXY","c":"Tcirq-handoff-rydberg_ccz_pulse2-codexcczsmoke0416","ccz_label":"rydberg_ccz_pulse2","custom_ccz_mode":"auto","d2":13,"f":3,"ghz_size":3,"latter_rounds":3,"noise":"uniform","p":0.001,"use_custom_ccz":true}
```

One more practical observation:

- the large `cczovernight0327d` shard files each contain `40000` shots, not the current script default of `200`
- so that dataset must have been produced with explicit shot-count overrides (`--num-sverr-shots` / `--num-shots-per-sverr`, or equivalent environment variables), not with the default CLI settings

## Session Update: April 17-20, 2026

I added and exercised a separate trajectory-based runner without modifying `MSC_foldedH/src/handoff_run_3err_ccz.py`.

New files created during this pass:

- `MSC_foldedH/src/handoff_trajectory_run.py`
- `MSC_foldedH/data/recreate_ccz_vs_f3.py`

### What `handoff_trajectory_run.py` Does

`MSC_foldedH/src/handoff_trajectory_run.py` is a copy of the custom-CCZ handoff runner with one key change in the exact-simulation prefix:

- it can treat the custom CCZ as a non-unitary Kraus channel
- and then sample that channel with `cirq.Simulator` in trajectory / Monte Carlo wavefunction mode instead of trying to store a full `20`-qubit density matrix

The high-level workflow is still:

1. run the Cirq exact-sim prefix,
2. collect accepted logical Bloch vectors by error location,
3. hand those distributions to the downstream Stim / `sinter` resampling stage,
4. emit a final `TaskStats` row.

This means:

- the non-unitary CCZ is applied directly only in the Cirq prefix
- the later Stim stage is still a stabilizer surrogate conditioned on the prefix outputs

### Why The Trajectory Method Was Added

The exact-sim prefix circuit has `20` qubits.

So:

- pure-state simulation scales like `2^20`
- density-matrix simulation scales like `4^20`

A full density matrix for that prefix is not practical in this environment.

The trajectory method avoids that blowup by sampling one Kraus branch per shot while keeping statevector memory scaling.

### Small Validation Against Exact Density-Matrix Evolution

Before using the trajectory runner on the full handoff prefix, I validated the channel machinery on a small `3`-qubit toy circuit built from the same custom CCZ Kraus gate.

I compared:

- the exact density-matrix evolution from `cirq.DensityMatrixSimulator`
- against the empirical average over `4000` trajectory samples from `cirq.Simulator`

Observed agreement:

- Frobenius-norm density-matrix error: about `1.74e-5`
- the logical Bloch-vector components agreed to the printed precision

Interpretation:

- the trajectory implementation itself appears numerically correct
- remaining uncertainty is mainly about the physical adequacy of the chosen channel model, not the Monte Carlo machinery

### First Production Trajectory Run With One Shared Custom CCZ

I first ran the trajectory workflow with one shared custom gate family, i.e. the same custom CCZ on all four CCZ placements.

Output directory:

- `MSC_foldedH/realcirqhandoffoutputs_rydberg_ccz_pulse2_trajectory_p1_trajfull0416a/`

Final aggregate:

- `100` shard files
- `4000000` total shots
- `1280544` discards
- `11182` logical errors
- zero-threshold expected attempts: about `1.47088`
- zero-threshold logical error rate: about `4.11185e-3`
- curve extent: about `2.768` expected attempts

That is the best full aggregate currently available in-repo for the single-family trajectory custom-CCZ run.

### Split Short-Range / Long-Range Gate Infrastructure

After that, I changed `handoff_trajectory_run.py` so the three short-range CCZ placements and the one long-range corner-to-corner CCZ placement can use different matrices.

The relevant input variables are now:

- `CCZ_gate_short_range`
- `CCZ_gate_long_range`

The three local CCZ placements use `CCZ_gate_short_range`.

The single opposite-corners CCZ placement uses `CCZ_gate_long_range`.

The runner metadata now also records:

- `short_range_ccz_label`
- `long_range_ccz_label`
- `short_range_ccz_gate_model`
- `long_range_ccz_gate_model`

### Measured Long-Range Gate

I replaced the placeholder long-range copy with the explicit long-range matrix provided during the session.

Contractivity check summary:

- short-range residual trace `Tr(I - K^\dagger K)`: about `6.98e-5`
- long-range residual trace `Tr(I - K^\dagger K)`: about `6.24e-2`

So the long-range gate is substantially more non-unitary / lossy than the short-range gate, which is consistent with the intended physical distinction.

### Full `f=3` Production Run With Measured Short/Long Gates

I then ran the full `f=3` production batch using:

- the short-range custom gate on the three local CCZ placements
- the measured long-range gate on the opposite-corners CCZ placement

Output directory:

- `MSC_foldedH/realcirqhandoffoutputs_sr-rydberg_ccz_pulse2_short_range__lr-rydberg_ccz_pulse2_long_range_trajectory_p1_actualgates_f3_0417a/`

Final aggregate:

- `100` shard files
- `4000000` total shots
- `1482197` discards
- `10373` logical errors
- zero-threshold expected attempts: about `1.58869`
- zero-threshold logical error rate: about `4.11986e-3`
- curve extent: about `2.991` expected attempts

Interpretation:

- compared to the single-family trajectory custom-CCZ run, the measured long-range gate clearly worsens acceptance
- the zero-threshold logical error rate stays in roughly the same range
- the main visible shift is in attempts / acceptance, which is what I would expect from a more strongly non-unitary long-range gate

### Plotting Infrastructure For The Trajectory Runs

`MSC_foldedH/data/recreate_ccz_vs_f3.py` was added to reconstruct the `ccz_vs_f3`-style comparison plot from:

- checked-in rows in `MSC_foldedH/data/all_paper_data_v2.csv`
- plus an aggregated trajectory shard directory

It plots:

- `|Y>`
- `|H_XY>` ideal CCZ
- a user-specified aggregated trajectory series
- `Ref. [9]`
- `Ref. [10]`

It also stops plotting once the finite-shot aggregate reaches zero observed failures, to avoid meaningless vertical drops to `LER = 0`.

Working command for the measured short/long-range `f=3` run:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC/MSC_foldedH
../.venv312/bin/python3 data/recreate_ccz_vs_f3.py \
  --traj-dir realcirqhandoffoutputs_sr-rydberg_ccz_pulse2_short_range__lr-rydberg_ccz_pulse2_long_range_trajectory_p1_actualgates_f3_0417a \
  --output /tmp/ccz_vs_f3_actualgates_f3_0417a.png \
  --traj-label '|H_XY> (Measured SR/LR CCZs)'
```

### Plot Comparison Notes

With the fully aggregated measured-gate `f=3` run:

- the measured SR/LR trajectory curve is now much closer in zero-threshold attempts to the historical ideal handoff row than the earlier single-family custom-CCZ trajectory run
- the measured-gate curve still only extends to about `3` expected attempts because `4e6` total shots is far smaller than the massive checked-in literature / historical rows
- so the current result is good enough for shape and zero-threshold comparison, but not for reproducing a very long ultra-low-LER tail

### Current Trajectory Bottom Line

At the end of this session, the trajectory work should be treated as:

- the trajectory implementation itself has passed a small exact density-matrix cross-check
- `handoff_trajectory_run.py` supports separate short-range and long-range CCZ gate inputs
- the single-family custom-CCZ trajectory run is complete and aggregated
- the measured short-range / long-range `f=3` trajectory run is complete and aggregated
- `recreate_ccz_vs_f3.py` is the current route to rebuild the comparison plot from those trajectory shard directories

## Session Update: April 21, 2026

I rechecked both the `ler_v2` plotting path and the standalone trajectory runner without changing repo code.

### What `recreate_ler_v2.py` Reproduces In The Current Environment

Working command:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC/MSC_foldedH
MPLCONFIGDIR=/tmp/mpl \
../.venv312/bin/python3 data/recreate_ler_v2.py \
  --csv data/all_paper_data_v2.csv \
  --output /tmp/ler_v2_codex.png
```

Observed result:

- `/tmp/ler_v2_codex.png` is byte-for-byte identical to `MSC_foldedH/data/ler_v2.recreated.png`
- it is **not** byte-for-byte or pixel-for-pixel identical to `MSC_foldedH/data/ler_v2.png`
- both images are `529 x 408`, but the pixel comparison against `data/ler_v2.png` gave:
  - maximum absolute channel difference: `1.0`
  - mean absolute channel difference: about `5.35e-2`

Interpretation:

- the checked-in reconstruction script is deterministic in the current `../.venv312` environment
- but what it currently recreates exactly is `ler_v2.recreated.png`, not the older checked-in `ler_v2.png`
- so the original `ler_v2.png` must have come from a different plotting script, plotting style, or matplotlib-version/rendering path than the one currently captured in `recreate_ler_v2.py`

### Full End-To-End Run Of `handoff_trajectory_run.py`

I also verified a real end-to-end single-shard run of `MSC_foldedH/src/handoff_trajectory_run.py` with the current code, using a temporary working directory so the script could write its normal outputs without touching this repo checkout.

Important runtime detail:

- the runner writes `realcirqhandoffmid_*` and `realcirqhandoffoutputs_*` relative to the current working directory, not relative to the script path
- so running from `/tmp/...` is sufficient to keep its normal artifacts out of `MSC_foldedH/`

Working command:

```bash
mkdir -p /tmp/msc_traj_verify
cd /tmp/msc_traj_verify
MPLCONFIGDIR=/tmp/mpl \
PYTHONPATH=/Users/colinstewart/Desktop/WORK/3Q-MSC/MSC_foldedH/src \
/Users/colinstewart/Desktop/WORK/3Q-MSC/.venv312/bin/python3 \
  /Users/colinstewart/Desktop/WORK/3Q-MSC/MSC_foldedH/src/handoff_trajectory_run.py \
  1 0 \
  --dfinal 13 \
  --tag codexverify0421 \
  --fault-distance-tag 3
```

This completed successfully with the script defaults:

- `num_sverr_shots = 10`
- `num_shots_per_sverr = 20`
- `sim_method = trajectory`
- `custom_ccz_mode = contractive_kraus`

Artifacts produced under `/tmp/msc_traj_verify/`:

- `realcirqhandoffmid_sr-rydberg_ccz_pulse2_short_range__lr-rydberg_ccz_pulse2_long_range_trajectory_p1_outscodexverify0421/bHXY_prepunitstab_dfinal13_0v2.json`
- `realcirqhandoffoutputs_sr-rydberg_ccz_pulse2_short_range__lr-rydberg_ccz_pulse2_long_range_trajectory_p1_codexverify0421/TaskStats_bHXYp1_prepunitstab_dfinal13_sr-rydberg_ccz_pulse2_short_range__lr-rydberg_ccz_pulse2_long_range_trajectory_0.json`

Final `TaskStats` row for that run:

- `shots=200`
- `errors=0`
- `discards=97`
- metadata label:
  - `Tcirq-handoff-sr-rydberg_ccz_pulse2_short_range__lr-rydberg_ccz_pulse2_long_range-trajectory-codexverify0421`

So as of April 21, 2026:

- `handoff_trajectory_run.py` does complete a full default single-shard run in the current environment
- the most reliable no-repo-write way to exercise it is to run it from a temp working directory with `PYTHONPATH` pointed at `MSC_foldedH/src`

## Session Update: April 22, 2026

I added a direct gate-count breakdown for the `handoff_trajectory_run.py` `|H_XY>` check workflow and also reran the `ccz_vs_f3`-style trajectory plotting path.

### Gate Breakdown For The `handoff_trajectory_run.py` Check Block

This section refers only to the `|H_XY>` exact-simulation prefix used by:

- `MSC_foldedH/src/handoff_trajectory_run.py`

and specifically to the two GHZ-based logical-check rounds that come from:

- `generate_sv_cirquit(...)`
- then the `CY`-moment rewrite inside `handoff_trajectory_run.py`

Important scope:

- this is **not** the whole handoff circuit
- it is just the check block in the exact-sim prefix
- counts below use the default path I inspected:
  - `prep="unitstab"`
  - `d2=13`
  - `p=0.001`
  - `sim_method="trajectory"`
  - `custom_ccz_mode="contractive_kraus"`

#### What The `CY` Check Becomes

In the original check primitive, each round has three `CY` moments with three `CY` gates each.

In `handoff_trajectory_run.py`, those are rewritten into:

- first check sublayer: `3` short-range `CCZ_kraus`
- second check sublayer: `3` `ControlledDiag`
- third check sublayer: `2` `ControlledDiag` plus `1` long-range `CCZ_kraus`

So per check round, the logical-check interaction itself is:

- `4` `CCZ_kraus`
- `5` `ControlledDiag`

#### Full One-Round Gate Totals

Including the surrounding GHZ prep / unprep, measurement, and inserted noise channels, one check round contains:

- `R x3`
- `RandomGateChannel x6`
- `H x2`
- `CX x4`
- `I x4`
- `CCZ_kraus x4`
- `ControlledDiag x5`
- `ZPow(-0.25) x1`
- `M x3`
- `DEPOLARIZE1q x39`
- `DEPOLARIZE2q x9`
- `DEPOLARIZE3q x4`

The workflow has two such rounds.

#### Per-Moment Table

For the inspected default circuit, round 1 occupies absolute moments `37-60` and round 2 occupies `61-84`.

| Relative moment in check round | Absolute moment round 1 | Absolute moment round 2 | Gates in that moment |
| --- | --- | --- | --- |
| 1 | 37 | 61 | `R x3` |
| 2 | 38 | 62 | `RandomGateChannel x3` |
| 3 | 39 | 63 | `H x1` |
| 4 | 40 | 64 | `DEPOLARIZE1q x3` |
| 5 | 41 | 65 | `CX x1`, `I x1` |
| 6 | 42 | 66 | `DEPOLARIZE2q x1`, `DEPOLARIZE1q x1` |
| 7 | 43 | 67 | `CX x1`, `I x1` |
| 8 | 44 | 68 | `DEPOLARIZE2q x1`, `DEPOLARIZE1q x1` |
| 9 | 45 | 69 | `CCZ_kraus x3` |
| 10 | 46 | 70 | `DEPOLARIZE3q x3`, `DEPOLARIZE1q x7` |
| 11 | 47 | 71 | `ControlledDiag x3` |
| 12 | 48 | 72 | `DEPOLARIZE2q x3`, `DEPOLARIZE1q x10` |
| 13 | 49 | 73 | `ControlledDiag x2`, `CCZ_kraus x1` |
| 14 | 50 | 74 | `DEPOLARIZE2q x2`, `DEPOLARIZE3q x1`, `DEPOLARIZE1q x9` |
| 15 | 51 | 75 | `CX x1`, `I x1` |
| 16 | 52 | 76 | `DEPOLARIZE2q x1`, `DEPOLARIZE1q x1` |
| 17 | 53 | 77 | `CX x1`, `I x1` |
| 18 | 54 | 78 | `DEPOLARIZE2q x1`, `DEPOLARIZE1q x1` |
| 19 | 55 | 79 | `ZPow(-0.25) x1` |
| 20 | 56 | 80 | `H x1` |
| 21 | 57 | 81 | `DEPOLARIZE1q x3` |
| 22 | 58 | 82 | `RandomGateChannel x3` |
| 23 | 59 | 83 | `M x3` |
| 24 | 60 | 84 | `DEPOLARIZE1q x3` |

### Fresh `ccz_vs_f3`-Style Trajectory Plot Recreation

I reran the checked-in plotting helper against the aggregated measured-gate trajectory shard directory:

```bash
cd /Users/colinstewart/Desktop/WORK/3Q-MSC/MSC_foldedH
MPLCONFIGDIR=/tmp/mpl \
../.venv312/bin/python3 data/recreate_ccz_vs_f3.py \
  --traj-dir realcirqhandoffoutputs_sr-rydberg_ccz_pulse2_short_range__lr-rydberg_ccz_pulse2_long_range_trajectory_p1_actualgates_f3_0417a \
  --output /tmp/ccz_vs_f3_codex.png \
  --traj-label '|H_XY> (Measured SR/LR CCZs)'
```

That command completed successfully and wrote:

- `/tmp/ccz_vs_f3_codex.png`

The trajectory aggregate used by that plot is:

- `shots=4000000`
- `errors=10373`
- `discards=1482197`
- metadata label:
  - `Tcirq-handoff-sr-rydberg_ccz_pulse2_short_range__lr-rydberg_ccz_pulse2_long_range-trajectory-actualgates_f3_0417a`

The plotted curve endpoints from that aggregate are:

- first plotted point:
  - expected attempts about `1.588687`
  - logical error rate about `4.119862e-3`
- last plotted point before the zero-failure finite-shot cutoff:
  - expected attempts about `2.525194`
  - logical error rate about `6.31e-7`

#### Relation To The Checked-In `data/ccz_vs_f3.png`

This fresh plotting path should be treated as a working recreation of a `ccz_vs_f3`-style comparison figure, not as a byte-for-byte reproduction of the checked-in PNG.

Observed difference:

- checked-in `data/ccz_vs_f3.png` sha1:
  - `c8f1fd9ee7ebba1b54347903455a676679f98620`
- fresh `/tmp/ccz_vs_f3_codex.png` sha1:
  - `d9164c427b1d0dfd056fc1b4b6e30f349ed93440`
- checked-in image shape:
  - `519 x 397`
- fresh image shape:
  - `529 x 408`

Interpretation:

- the current script does successfully regenerate the intended comparison plot from the current CSV plus trajectory shards
- but the checked-in `data/ccz_vs_f3.png` was rendered with a different plotting script, figure size, or matplotlib/rendering environment than the current `recreate_ccz_vs_f3.py`
