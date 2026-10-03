"""Final diagnostics and report callback must work before post-evaluation aggregation."""
import json

from experiments.nest_clip_v1.balanced_view_relation_500_v1.report import write_report, METRICS, DATASETS
from experiments.nest_clip_v1.balanced_view_relation_500_v1.run import summarize_relation


def test_diagnostic_snapshots_and_last50_use_raw_values():
    records=[]
    for step in range(1,501):
        row={key:step/1000 for key in METRICS if not key.endswith('cosine_preference')}
        row.update(step=step,c_PP_mean=.8,c_RP_mean=.3,c_RR_mean=.7,c_PR_mean=.4)
        records.append(row)
    result=summarize_relation(records)
    assert result['1']['inc']==.001 and result['500']['inc']==.5
    assert result['last50']['inc']==.4755
    assert result['last50']['P_cosine_preference']==.5
    assert abs(result['last50']['R_cosine_preference']-.3)<1e-12


def test_native_callback_without_summary_does_not_raise(tmp_path):
    metrics={ds:{dr:{f'R@{k}':.5 for k in (1,5,10)} for dr in ('I2T','T2I')} for ds in DATASETS}
    record=dict(scores=dict(Score5_R1=.5,J_long3=.5,J_long=.5),metrics=metrics,
                checkpoint_sha256='full',bare_sha256='bare')
    state=dict(status='running',baseline=record,relation_trial='trial',trials={'trial':{'budgets':{'500':record}}})
    write_report(state,tmp_path)
    assert 'Pending final diagnostic aggregation.' in (tmp_path/'VIEW_RELATION_500_REPORT.md').read_text()
    assert json.loads((tmp_path/'RESULTS.json').read_text())['status']=='running'
