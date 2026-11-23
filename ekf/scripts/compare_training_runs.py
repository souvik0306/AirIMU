"""Compare validation-selected checkpoints without reading held-out test data."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

from ekf.gradients import summarize_batches


def summarize(run):
    run = Path(run)
    epochs = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
    best = min(epochs, key=lambda row: row["validation_xy_mse"])
    flights = [json.loads(line) for line in (run/"flights.jsonl").read_text().splitlines()]
    selected = [row for row in flights if row["phase"] == "validation" and row["epoch"] == best["epoch"]]
    if not selected:
        raise ValueError(f"No selected validation flight metrics in {run}")
    metadata=json.loads((run/'eskf_metadata.json').read_text())
    initial=json.loads((run/'initial_validation.json').read_text()) if (run/'initial_validation.json').exists() else None
    batches=[json.loads(line) for line in (run/'batches.jsonl').read_text().splitlines()] if (run/'batches.jsonl').exists() else []
    gyro_summary={}
    for name,key in (('flight_mean_one_sigma_coverage','one_sigma_coverage'),
                     ('flight_mean_normalized_residual_square','normalized_residual_square')):
        available=[row['gyro_diagnostics'][key] for row in selected if key in row['gyro_diagnostics']]
        if available:gyro_summary[name]=[sum(v[axis] for v in available)/len(available) for axis in range(3)]
    for name,key in (('median_of_flight_medians','median'),('median_of_flight_p95','p95')):
        available=[row['gyro_diagnostics'][key] for row in selected if key in row['gyro_diagnostics']]
        if available:gyro_summary[name]=[median(v[axis] for v in available) for axis in range(3)]
    return {"run": str(run), "best_epoch": best["epoch"], 'configuration':metadata.get('configuration'),
            'epochs_completed':len(epochs), 'initial_validation':initial,
            "validation_xy_rmse": math.sqrt(best["validation_xy_mse"]),
            'validation_xy_mse':best['validation_xy_mse'],
            'validation_covariance':best['validation_covariance'],
            'validation_gyro_calibration_loss':sum(row['gyro_calibration_loss'] for row in selected)/len(selected),
            'validation_gyro_diagnostics':gyro_summary,
            'gradient_summary':summarize_batches([row for row in batches if row['phase']=='train']),
            "training_seconds": sum(row["epoch_seconds"] for row in epochs),
            "main_peak_memory_mb": max(row.get("main_peak_memory_mb", 0.) for row in flights),
            "worker_peak_memory_mb": max((row.get("worker_peak_memory_mb") or 0.) for row in flights),
            "gpu_peak_memory_mb": max(row.get("gpu_peak_memory_mb", 0.) for row in flights),
            "flights": {row["flight"]: {"xy_rmse": row["validation_xy_rmse"],
                         "gyro_calibration_loss": row["gyro_calibration_loss"],
                         "smoothness_loss": row["smoothness_loss"],
                         "gyro": row["gyro_diagnostics"],
                         'reference_quality':row.get('gyro_reference_quality'),
                         "correction_drift_rmse": row["correction_drift_rmse"]} for row in selected}}


def validate_matched_ab(runs, expected_epochs):
    metadata=[json.loads((Path(run)/'eskf_metadata.json').read_text()) for run in runs]
    if len(runs)!=2 or [row['configuration'] for row in metadata]!=['A','B']:
        raise ValueError('Provide exactly configuration A and B, in that order.')
    keys=('initial_checkpoint','initial_checkpoint_sha256','config','flights','seed','loss_options',
          'height_updates','height_measurement_std','covariance_policy','flight_batch_size',
          'gradient_clip','correction_weight','replay_backend','replay_workers','feature_cache_mb',
          'gyro_reference_quality_policy','evaluate_initial')
    for key in keys:
        if metadata[0].get(key)!=metadata[1].get(key):
            raise ValueError(f'Matched A/B metadata differs in {key}.')
    groups=[{group['name']:group['lr'] for group in row['optimizer_groups']} for row in metadata]
    if set(groups[0])-set(groups[1]) or set(groups[1])-set(groups[0])!={'gru2'}:
        raise ValueError('Configuration B must add only the GRU2 optimizer group.')
    if any(groups[0].get(name)!=groups[1].get(name) for name in groups[0]):
        raise ValueError('Matched covariance-head/bias learning rates differ.')
    for run in runs:
        epochs=[json.loads(line) for line in (Path(run)/'metrics.jsonl').read_text().splitlines()]
        if [row['epoch'] for row in epochs]!=list(range(1,expected_epochs+1)):
            raise ValueError(f'{run} does not contain all {expected_epochs} epochs.')
        initial=json.loads((Path(run)/'initial_validation.json').read_text())
        if initial.get('optimizer_steps')!=0 or not initial.get('weights_unchanged'):
            raise ValueError('Epoch-zero validation must use unchanged initial weights.')


def paired_report(records):
    a,b=records
    initial=a['initial_validation']
    if initial is None or b['initial_validation'] is None:
        raise ValueError('Matched comparison requires initial validation for both runs.')
    if not math.isclose(initial['validation_xy_mse'],b['initial_validation']['validation_xy_mse'],rel_tol=1e-7,abs_tol=1e-9):
        raise ValueError('A and B did not reproduce the same initial validation baseline.')
    paired=[]
    initial_path=Path(a['run'])/'initial_validation_flights.jsonl'
    baseline={row['flight']:row['validation_xy_rmse'] for row in
              (json.loads(line) for line in initial_path.read_text().splitlines())} if initial_path.exists() else {}
    for flight in sorted(a['flights']):
        ar=a['flights'][flight]['xy_rmse'];br=b['flights'][flight]['xy_rmse']
        if a['flights'][flight]['reference_quality']!=b['flights'][flight]['reference_quality']:
            raise ValueError(f'A/B reference masks differ for {flight}.')
        row={'flight':flight,'a_xy_rmse':ar,'b_xy_rmse':br,
                       'b_rmse_improvement_percent_vs_a':100*(1-br/ar),
                       'b_mse_reduction_vs_a':ar*ar-br*br}
        if baseline:
            row.update(initial_xy_rmse=baseline[flight],
                a_rmse_improvement_percent_vs_initial=100*(1-ar/baseline[flight]),
                b_rmse_improvement_percent_vs_initial=100*(1-br/baseline[flight]))
        paired.append(row)
    for record in records:
        record['validation_rmse_improvement_percent_vs_initial']=100*(1-record['validation_xy_rmse']/initial['validation_xy_rmse'])
    return {'runs':records,'initial_validation_xy_rmse':initial['validation_xy_rmse'],
            'b_rmse_improvement_percent_vs_a':100*(1-b['validation_xy_rmse']/a['validation_xy_rmse']),
            'b_better_flights':sum(row['b_xy_rmse']<row['a_xy_rmse'] for row in paired),
            'a_better_flights':sum(row['a_xy_rmse']<row['b_xy_rmse'] for row in paired),
            'paired_flights':paired,
            'interpretation':'Validation-selected, single-seed A/B comparison with matched corrected references; held-out test flights are not evaluated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument('--require-matched-ab',action='store_true')
    parser.add_argument('--expected-epochs',type=int,default=25)
    parser.add_argument('--plots',action='store_true')
    args = parser.parse_args()
    if args.require_matched_ab:
        validate_matched_ab(args.runs,args.expected_epochs)
    records = [summarize(run) for run in args.runs]
    ids = set(records[0]["flights"])
    if any(set(row["flights"]) != ids for row in records):
        raise ValueError("Comparisons require identical validation flights.")
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    report=paired_report(records) if args.require_matched_ab else records
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.require_matched_ab:
        paired=report['paired_flights']
        with output.with_name(output.stem+'_flights.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(paired[0]))
            writer.writeheader();writer.writerows(paired)
        curves=[]
        for record in records:
            for line in (Path(record['run'])/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                curves.append({'configuration':record['configuration'],'epoch':row['epoch'],
                    'validation_xy_rmse':row['validation_xy_rmse'],
                    'train_xy_rmse':row['train_xy_rmse'],'epoch_seconds':row['epoch_seconds'],
                    'clip_fraction':row.get('train_gradients',{}).get('clip_fraction')})
        with output.with_name(output.stem+'_epochs.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(curves[0]))
            writer.writeheader();writer.writerows(curves)
        if args.plots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            for record in records:
                values=[row for row in curves if row['configuration']==record['configuration']]
                axes[0,0].plot([row['epoch'] for row in values],[row['validation_xy_rmse'] for row in values],label=record['configuration'])
                axes[1,0].plot([row['epoch'] for row in values],[row['clip_fraction'] for row in values],label=record['configuration'])
                q=record['validation_covariance']['gyro_cov']['mean']
                axes[0,1].plot(list('XYZ'),[math.sqrt(v) for v in q],marker='o',label=record['configuration'])
            initial=records[0]['initial_validation']
            axes[0,0].axhline(initial['validation_xy_rmse'],linestyle='--',color='gray',label='Initial')
            axes[0,0].set(xlabel='Epoch',ylabel='Validation XY RMSE (m/s)')
            axes[0,1].set(ylabel='Gyro sqrt(mean variance) (rad/s)')
            axes[1,0].set(xlabel='Epoch',ylabel='Clipped batch fraction',ylim=(0,1.05))
            axes[1,1].scatter([row['a_xy_rmse'] for row in paired],[row['b_xy_rmse'] for row in paired])
            maximum=max(max(row['a_xy_rmse'],row['b_xy_rmse']) for row in paired)
            axes[1,1].plot([0,maximum],[0,maximum],linestyle='--',color='gray')
            axes[1,1].set(xlabel='A flight RMSE (m/s)',ylabel='B flight RMSE (m/s)')
            for axis in axes.flat:
                axis.grid(alpha=.2)
                if axis.get_legend_handles_labels()[0]:axis.legend()
            fig.savefig(output.with_suffix('.png'),dpi=180)
            plt.close(fig)
    for row in records:
        print(f"{row['run']}: epoch={row['best_epoch']} validation_xy_rmse={row['validation_xy_rmse']:.6g} "
              f"seconds={row['training_seconds']:.1f}")


if __name__ == "__main__":
    main()
