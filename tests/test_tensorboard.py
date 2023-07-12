"""
Unit tests for tensorboard.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

from ml4ptp.tensorboard import CustomTensorBoardLogger


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__custom_tensorboard_logger(tmp_path: Path) -> None:

    logger = CustomTensorBoardLogger(save_dir=tmp_path, name='test')
    logger.log_metrics({'test': 1.0}, step=0)

    assert logger.name == 'test'
    assert logger.version == 0
    assert logger.save_dir == tmp_path.as_posix()
    assert logger.log_dir == (tmp_path / 'test' / 'version_0').as_posix()
