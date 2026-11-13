# Final source-calendar numerical sweep

**Status: FINAL_SOURCE_ONLY_INDEPENDENT_REPLAY.**

Inputs: normalized `approved_model.json` and `hardware_model.json` only. No NCU/event/target timing files were inspected.
Unknown tails have explicit zero numeric service cost, while every source dependency remains in the schedule.
Each node receives `ceil(demand/rate)` cycles; it starts at the maximum predecessor finish and previous finish on that resource.
The 19 normalized source calendars are each reported below and checked against their regenerated expected-cycle values.

Rates use FP16/TF32 official dense peaks divided by `SM*f`; HBM uses `7.7e12/(f*HBM_divisor)` bytes per SM cycle. The computed FP16, TF32 and divisor-148 HBM rates are cross-checked against `hardware_model.json`.

## Full kernel sweep

| Kernel | Config | Calendar | Grid | Residency | Waves | CTA span (cy) | Kernel (cy) | Time (µs) | HBM bytes/kernel | FP16 FLOPs/kernel | TF32 FLOPs/kernel |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | grid2048 | K1_batch_1..8 | 2048 | 8 | 2 | 2242 | 30879 | 15.714504 | 50872320 | 2516582400 | 0 |
| K1 | grid4096 | K1_batch_1..8 | 4096 | 8 | 4 | 2242 | 61753 | 31.426463 | 101744640 | 5033164800 | 0 |
| K1 | grid8192 | K1_batch_1..8 | 8192 | 8 | 7 | 2242 | 123487 | 62.843257 | 203489280 | 10066329600 | 0 |
| K2 | L16 | K2_16 | 128 | 1 | 1 | 46619 | 46619 | 23.724682 | 101188608 | 19058917376 | 0 |
| K2 | L32 | K2_32 | 128 | 1 | 1 | 83643 | 83643 | 42.566412 | 185598976 | 38386270208 | 0 |
| K2 | L64 | K2_64 | 128 | 1 | 1 | 157691 | 157691 | 80.249873 | 354419712 | 77040975872 | 0 |
| K3P8 | L8 | K3_8 | 32 | 1 | 1 | 50203 | 50203 | 25.548601 | 37748736 | 0 | 536870912 |
| K3P8 | L8_HBMdiv32 | K3_8_div32 | 32 | 1 | 1 | 15371 | 15371 | 7.822392 | 37748736 | 0 | 536870912 |
| K3P8 | L8_etaTF32_0.5 | K3_8_etaHalf | 32 | 1 | 1 | 50757 | 50757 | 25.830534 | 37748736 | 0 | 536870912 |
| K4 | L16 | K4_16 | 128 | 1 | 1 | 57560 | 57560 | 29.292621 | 159907840 | 19327352832 | 0 |
| K4 | L16_HBMdiv128 | K4_16_div128 | 128 | 1 | 1 | 56032 | 56032 | 28.515013 | 159907840 | 19327352832 | 0 |
| K4 | L16_eta64_0.5 | K4_16_etaHalf | 128 | 1 | 1 | 58453 | 58453 | 29.747074 | 159907840 | 19327352832 | 0 |
| K4 | L32 | K4_32 | 128 | 1 | 1 | 107032 | 107032 | 54.469211 | 311427072 | 38654705664 | 0 |
| K4 | L64 | K4_64 | 128 | 1 | 1 | 205976 | 205976 | 104.822392 | 614465536 | 77309411328 | 0 |

Wave interpretation: K1 uses `q=ceil(grid/148)` and batches cohorts of up to eight CTAs per SM; its total is `max(floor(q/8)*Cbatch8 + Cbatch(q mod 8), ceil(grid*24840*f/7.7e12))`. K2, K4, and K3 P8 use their instructed one balanced wave, with kernel cycles equal to the one-CTA span. Sensitivity calendars are displayed separately.

## Calendar audit

Resource-work floor is the maximum sum of service cycles for any one resource in the calendar; DAG span additionally includes dependency-caused idle gaps.

| Calendar | Resource-work floor (cy) | DAG span (cy) | Expected (cy) | Delta | HBM bytes | SMEM bytes | TMEM bytes | FP16/TC16 FLOPs | TF32 FLOPs | SIMT results | CVT elements | ISSUE warp-inst | Control nodes / demand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K1_batch_1 | 940 | 2242 | 2242 | 0 | 24840 | 116864 | 0 | 1228800 | 0 | 13152 | 14336 | 42 | 3 / 0 |
| K1_batch_2 | 1877 | 4426 | 4426 | 0 | 49680 | 233728 | 0 | 2457600 | 0 | 26304 | 28672 | 84 | 3 / 0 |
| K1_batch_3 | 2816 | 6643 | 6643 | 0 | 74520 | 350592 | 0 | 3686400 | 0 | 39456 | 43008 | 126 | 3 / 0 |
| K1_batch_4 | 3754 | 8830 | 8830 | 0 | 99360 | 467456 | 0 | 4915200 | 0 | 52608 | 57344 | 168 | 3 / 0 |
| K1_batch_5 | 4693 | 11049 | 11049 | 0 | 124200 | 584320 | 0 | 6144000 | 0 | 65760 | 71680 | 210 | 3 / 0 |
| K1_batch_6 | 5630 | 13238 | 13238 | 0 | 149040 | 701184 | 0 | 7372800 | 0 | 78912 | 86016 | 252 | 3 / 0 |
| K1_batch_7 | 6569 | 15456 | 15456 | 0 | 173880 | 818048 | 0 | 8601600 | 0 | 92064 | 100352 | 294 | 3 / 0 |
| K1_batch_8 | 7507 | 17641 | 17641 | 0 | 198720 | 934912 | 0 | 9830400 | 0 | 105216 | 114688 | 336 | 3 / 0 |
| K3_8 | 44592 | 50203 | 50203 | 0 | 1179648 | 1048576 | 1146880 | 0 | 16777216 | 0 | 65536 | 0 | 33 / 0 |
| K3_8_div32 | 9648 | 15371 | 15371 | 0 | 1179648 | 1048576 | 1146880 | 0 | 16777216 | 0 | 65536 | 0 | 33 / 0 |
| K3_8_etaHalf | 44592 | 50757 | 50757 | 0 | 1179648 | 1048576 | 1146880 | 0 | 16777216 | 0 | 65536 | 0 | 33 / 0 |
| K4_16 | 47212 | 57560 | 57560 | 0 | 1249280 | 3702784 | 6356992 | 150994944 | 0 | 1317408 | 983040 | 0 | 355 / 0 |
| K4_32 | 91948 | 107032 | 107032 | 0 | 2433024 | 7405568 | 12648448 | 301989888 | 0 | 2634816 | 1966080 | 0 | 707 / 0 |
| K4_64 | 181420 | 205976 | 205976 | 0 | 4800512 | 14811136 | 25231360 | 603979776 | 0 | 5269632 | 3932160 | 0 | 1411 / 0 |
| K4_16_div128 | 40880 | 56032 | 56032 | 0 | 1249280 | 3702784 | 6356992 | 150994944 | 0 | 1317408 | 983040 | 0 | 355 / 0 |
| K4_16_etaHalf | 47212 | 58453 | 58453 | 0 | 1249280 | 3702784 | 6356992 | 150994944 | 0 | 1317408 | 983040 | 0 | 355 / 0 |
| K2_16 | 29881 | 46619 | 46619 | 0 | 790536 | 1740800 | 9551872 | 148897792 | 0 | 794144 | 1024000 | 0 | 785 / 0 |
| K2_32 | 54809 | 83643 | 83643 | 0 | 1449992 | 3481600 | 18989056 | 299892736 | 0 | 1588288 | 2072576 | 0 | 1569 / 0 |
| K2_64 | 104665 | 157691 | 157691 | 0 | 2768904 | 6963200 | 37863424 | 601882624 | 0 | 3176576 | 4169728 | 0 | 3137 / 0 |

## Trend readout

- K2 L16/32/64 cycles: 46619 / 83643 / 157691; time (µs): 23.724682 / 42.566412 / 80.249873.
- K4 L16/32/64 cycles: 57560 / 107032 / 205976; time (µs): 29.292621 / 54.469211 / 104.822392.
- K1 grids 2048/4096/8192 cycles: 30879 / 61753 / 123487; time (µs): 15.714504 / 31.426463 / 62.843257.

## Unreconciled differences

All 19 independently recomputed calendar spans match the regenerated approved expected-cycle values; there are no numeric disagreements.

## Full per-kernel resource totals

Demands below include total work across each reported grid; K1 uses the per-CTA calendar demands multiplied by grid. `control` demand remains zero and all control-node counts/demand-by-unit are retained in the JSON calendar records.

| Kernel | Config | HBM bytes | SMEM bytes | TMEM bytes | HMMA FLOPs | TC16 FLOPs | TF32 FLOPs | SIMT results | CVT elements | ISSUE warp-inst | Control demand |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | grid2048 | 50872320 | 239337472 | 0 | 2516582400 | 0 | 0 | 26935296 | 29360128 | 86016 | 0 |
| K1 | grid4096 | 101744640 | 478674944 | 0 | 5033164800 | 0 | 0 | 53870592 | 58720256 | 172032 | 0 |
| K1 | grid8192 | 203489280 | 957349888 | 0 | 10066329600 | 0 | 0 | 107741184 | 117440512 | 344064 | 0 |
| K2 | L16 | 101188608 | 222822400 | 1222639616 | 0 | 19058917376 | 0 | 101650432 | 131072000 | 0 | 0 |
| K2 | L32 | 185598976 | 445644800 | 2430599168 | 0 | 38386270208 | 0 | 203300864 | 265289728 | 0 | 0 |
| K2 | L64 | 354419712 | 891289600 | 4846518272 | 0 | 77040975872 | 0 | 406601728 | 533725184 | 0 | 0 |
| K3P8 | L8 | 37748736 | 33554432 | 36700160 | 0 | 0 | 536870912 | 0 | 2097152 | 0 | 0 |
| K3P8 | L8_HBMdiv32 | 37748736 | 33554432 | 36700160 | 0 | 0 | 536870912 | 0 | 2097152 | 0 | 0 |
| K3P8 | L8_etaTF32_0.5 | 37748736 | 33554432 | 36700160 | 0 | 0 | 536870912 | 0 | 2097152 | 0 | 0 |
| K4 | L16 | 159907840 | 473956352 | 813694976 | 0 | 19327352832 | 0 | 168628224 | 125829120 | 0 | 0 |
| K4 | L16_HBMdiv128 | 159907840 | 473956352 | 813694976 | 0 | 19327352832 | 0 | 168628224 | 125829120 | 0 | 0 |
| K4 | L16_eta64_0.5 | 159907840 | 473956352 | 813694976 | 0 | 19327352832 | 0 | 168628224 | 125829120 | 0 | 0 |
| K4 | L32 | 311427072 | 947912704 | 1619001344 | 0 | 38654705664 | 0 | 337256448 | 251658240 | 0 | 0 |
| K4 | L64 | 614465536 | 1895825408 | 3229614080 | 0 | 77309411328 | 0 | 674512896 | 503316480 | 0 | 0 |

Unknown latency/issue/SFU/barrier tails remain unpriced at zero numeric cost, so these are explicit ideal-tail source schedules, not measured runtimes or lower-bound claims. Source-level review issues from `approved_model.json` remain recorded in the JSON output.
