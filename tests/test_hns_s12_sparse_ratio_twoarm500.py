import math
import subprocess
import sys

from recovery import hns_s12_sparse_ratio_twoarm500 as r


def test_requested_ratios_have_equal_mass_and_coefficients():
    assert r.ARMS['E1-AlignMatched']['sparsity_weights']==[2.25,2.25,.5]
    assert all(math.isclose(x,5/3,abs_tol=1e-12) for x in r.ARMS['E2-Uniform']['sparsity_weights'])
    for spec in r.ARMS.values():
        assert math.isclose(sum(spec['sparsity_weights']),5.,abs_tol=1e-12)


def test_commands_are_fresh_common0_and_local_only():
    r.configure()
    for arm in r.ARMS:
        for smoke in (True,False):
            cmd=r.runner.training_command(arm,smoke)
            assert '--resume' not in cmd
            assert cmd[cmd.index('--max-updates')+1]==('5' if smoke else '500')
            assert cmd[cmd.index('--image-root')+1]=='/root/said_s02_stage500/ShareGPT4V'
            assert cmd[cmd.index('-m')+1]==r.ENTRY


def test_equivalence_receipt_subprocess(tmp_path):
    out=tmp_path/'equivalence.json'
    subprocess.run([sys.executable,'-m','recovery.hns_sparse_ratio_equivalence','--output',str(out)],check=True)
    assert out.exists()
