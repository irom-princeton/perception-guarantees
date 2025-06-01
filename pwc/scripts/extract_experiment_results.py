import hydra
import argparse
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main(cfg):
    """
    Main function to extract results.
    """
    experiment = instantiate(cfg.experiment)
    experiment.extract_results()



if __name__ == "__main__":
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str, 
        required=False,
        default="pwc-3detr",
        help="Name of the config file"
    )
    
    # parse arguments
    args = parser.parse_args()
    
    # load the configs from file
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=f"../configs")
    cfg = hydra.compose(config_name=args.config_name)

    # run the main function
    main(
        cfg=cfg
    )