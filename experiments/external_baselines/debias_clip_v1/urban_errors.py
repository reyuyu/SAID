"""Requested Urban correctness categories and unbiased top-k agreement audit."""
import json

import torch

from .common import RUN, dump


def main():
    said = torch.load(RUN/'said-urban-predictions.pt', weights_only=True)
    debias = torch.load(RUN/'unified/urban-embeddings.pt', weights_only=True)
    names = said['image_basenames']
    assert names == debias['image_basenames'] and len(names) == 1000
    error, ranking = {}, {}
    for direction, key in [('I2T', 'similarity_i2t'), ('T2I', 'similarity_t2i')]:
        # Use GPU top-k from each original environment; CPU/tie behavior must
        # not silently replace the frozen evaluator's actual predictions.
        s = said[key+'_top10']
        d = debias[key+'_top10']
        s1, d1 = said[key+'_top1'], debias[key+'_top1']
        s5, d5 = said[key+'_top5'], debias[key+'_top5']
        groups = {key: [] for key in ['both_correct', 'said_only_correct', 'debias_only_correct', 'both_wrong']}
        per_query = []
        agreements = []
        overlap5, overlap10 = [], []
        for i, name in enumerate(names):
            sc, dc = int(s1[i]) == i, int(d1[i]) == i
            group = ('both_correct' if sc and dc else 'said_only_correct' if sc else
                     'debias_only_correct' if dc else 'both_wrong')
            groups[group].append(name)
            a, b = s[i].tolist(), d[i].tolist()
            agreements.append(int(s1[i]) == int(d1[i]))
            overlap5.append(len(set(s5[i].tolist()) & set(d5[i].tolist()))/5)
            overlap10.append(len(set(a) & set(b))/10)
            per_query.append({'query_id': name, 'category': group,
                'said_top1_id': names[int(s1[i])], 'debias_top1_id': names[int(d1[i])],
                'said_top10_ids': [names[j] for j in a], 'debias_top10_ids': [names[j] for j in b]})
        error[direction] = {'counts': {key: len(value) for key, value in groups.items()},
                            'sample_ids': groups, 'n_queries': 1000,
                            'said_correct_queries': len(groups['both_correct'])+len(groups['said_only_correct']),
                            'debias_correct_queries': len(groups['both_correct'])+len(groups['debias_only_correct'])}
        ranking[direction] = {'top1_agreement_count': sum(agreements), 'top1_agreement_rate': sum(agreements)/1000,
            'mean_top5_candidate_overlap': sum(overlap5)/1000,
            'mean_top10_candidate_overlap': sum(overlap10)/1000, 'per_query_top10': per_query}
        assert sum(error[direction]['counts'].values()) == 1000
    dump(RUN/'urban-error-analysis.json', {'status': 'COMPLETE', 'directions': error,
        'diagnostic_only': True, 'no_manual_query_selection': True, 'no_metric_or_model_changes': True})
    dump(RUN/'urban-ranking-agreement.json', {'status': 'COMPLETE', 'directions': ranking,
        'spearman': None, 'spearman_note': 'Optional full-rank correlation omitted; every query top10 is preserved.'})
    print(json.dumps({'counts': {d: v['counts'] for d, v in error.items()},
                     'top1_agreement': {d: v['top1_agreement_rate'] for d, v in ranking.items()}}), flush=True)


if __name__ == '__main__':
    main()
