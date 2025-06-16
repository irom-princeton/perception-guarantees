import hydra
import argparse
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main(cfg):
    """
    Main function to run the evaluation script.
    """
    perception_model = instantiate(cfg.perception)
    calibration_method = instantiate(cfg.calibration)
    env = instantiate(cfg.nav_sim)
    planner = instantiate(cfg.planning)
    experiment = instantiate(cfg.experiment)

    experiment.run(
        env=env,
        perception_model=perception_model,
        calibration_method=calibration_method,
        planner=planner,
    )


if __name__ == "__main__":
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str, 
        required=False,
        default= "pwc-numcc", # "pwc-3detr",
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