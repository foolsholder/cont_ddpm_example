from default_cifar_config import create_default_cifar_config
from diffusion import DiffusionRunner

config = create_default_cifar_config()
diffusion = DiffusionRunner(config)

config.checkpoints_prefix = ''

diffusion.train(
    project_name='vp-diffusion',
    experiment_name='vp-sde'
)
