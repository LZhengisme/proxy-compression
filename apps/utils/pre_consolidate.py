from dataclasses import dataclass
import logging
from pathlib import Path
from omegaconf import OmegaConf
from contextlib import ExitStack

import torch

from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.logger import init_logger

logger = logging.getLogger()


@dataclass
class ConsolidationArgs:
    ckpt_dir: str = ""
    seed: int = 42

def main():
    init_logger()
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(ConsolidationArgs())
    cfg = OmegaConf.merge(default_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    with ExitStack() as context_stack:
        torch.manual_seed(cfg.seed)
        if (
            Path(cfg.ckpt_dir).exists()
            and (Path(cfg.ckpt_dir) / "params.json").exists()
            and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
        ):
            consolidate_path = Path(cfg.ckpt_dir)
        else:
            consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
            if not consolidate_path.exists():
                logger.info(f"Start consolidating distributed checkpoints in {cfg.ckpt_dir}.")
                consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)
            else:
                logger.info(f"{cfg.ckpt_dir} is already consolidated in {str(consolidate_path)}.")

if __name__ == "__main__":
    main()
