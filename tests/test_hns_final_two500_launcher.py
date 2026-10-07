"""Exercise real torchrun argparse, not just a guessed command string."""
import ast
from torch.distributed.run import parse_args
from recovery.hns_final_two500_continue import safe_torchrun,ROOT


def test_real_torchrun_parser_and_worker_argument_identity():
    args=['--run','/tmp/immutable-run','--experiment','/tmp/report']
    command=safe_torchrun('recovery.hns_gradient_audit',*args)
    parsed=parse_args(command[1:])
    assert parsed.no_python and parsed.training_script==str(ROOT/'.venv/bin/python')
    assert parsed.training_script_args[0]=='-c'
    code=ast.parse(parsed.training_script_args[1])
    assignment=next(n for n in code.body if isinstance(n,ast.Assign))
    assert ast.literal_eval(assignment.value)==['audit',*args]
    assert '--run' not in command


def test_training_command_unchanged():
    from recovery.hns_final_two500_continue import ORIGINAL_TORCHRUN
    args=['--arm','HNS-InnerFocus','--worker','--max-updates','500']
    assert safe_torchrun('recovery.hns_final_two500',*args)==ORIGINAL_TORCHRUN('recovery.hns_final_two500',*args)
