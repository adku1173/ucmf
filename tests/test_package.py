"""The extracted package must not import the original experiment stack."""

import importlib.util
import subprocess
import sys

from ucmf import optim


def test_imports_are_independent_of_fa2026_and_acoustic_packages():
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; '
            'from ucmf import ISTA, FISTA; '
            'from ucmf.optim import ISTA as OptimISTA, FISTA as OptimFISTA; '
            'assert (ISTA, FISTA) == (OptimISTA, OptimFISTA); '
            'forbidden = ("fa2026", "acoular", "tensorflow", "matplotlib", "h5py", "irdl", "ray"); '
            'assert not any(name.split(".")[0] in forbidden for name in sys.modules)',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_removed_state_framework_is_not_part_of_the_package_interface():
    assert not hasattr(optim, 'IterationState')
    assert not hasattr(optim, 'IterationContext')
    assert importlib.util.find_spec('ucmf.optim.iteration') is None
