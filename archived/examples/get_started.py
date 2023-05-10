import numpy as np
import pypolo
env = pypolo.experiments.environments.get_environment("N17E073", "./data/srtm/")

# [xmin, xmax, ymin, ymax]
env_extent = [-11.0, 11.0, -11.0, 11.0]
task_extent = [-10.0, 10.0, -10.0, 10.0]

sensing_rate = 1.0
noise_scale = 1.0
sensor = pypolo.sensors.Ranger(
    rate=sensing_rate,  # collects 1 sonar measurement per sec
    env=env,  # the sensor samples from this environment
    env_extent=env_extent,  # workspace extent
    noise_scale=noise_scale,  # observational Gaussian noise scale (i.e., standard deviation)
)

seed = 0
num_init_samples = 50
rng = pypolo.experiments.utilities.seed_everything(seed)

bezier = pypolo.strategies.Bezier(task_extent=task_extent, rng=rng)
x_init = bezier.get(num_states=num_init_samples)
y_init = sensor.sense(states=x_init, rng=rng).reshape(-1, 1)

control_rate = 10
max_lin_vel = 1.0
tolerance = 0.1

robot = pypolo.robots.USV(
        init_state=np.array([x_init[-1, 0], x_init[-1, 1], np.pi / 2]),
        control_rate=control_rate,
        max_lin_vel=max_lin_vel,
        tolerance=tolerance,
        sampling_rate=sensing_rate,
    )

init_amplitude = 1.0
init_lengthscale = 0.5
init_noise = 1.0

lr_hyper = 0.01
lr_nn = 0.001

kernel = pypolo.kernels.RBF(amplitude=init_amplitude, lengthscale=init_lengthscale)

model = pypolo.models.GPR(
    x_train=x_init,
    y_train=y_init,
    kernel=kernel,
    noise=init_noise,
    lr_hyper=lr_hyper,
    lr_nn=lr_nn,
)

model.optimize(num_iter=model.num_train, verbose=True)

eval_grid = [50, 50]

evaluator = pypolo.experiments.Evaluator(
        sensor=sensor,
        task_extent=task_extent,
        eval_grid=eval_grid,
    )

logger = pypolo.experiments.Logger(evaluator.eval_outputs)

mean, std, error = evaluator.eval_prediction(model)
logger.append(mean, std, error, x_init, y_init, model.num_train)

num_candidates = 1000

strategy = pypolo.strategies.MyopicPlanning(
            task_extent=task_extent,
            rng=rng,
            num_candidates=num_candidates,
            robot=robot,
        )

max_num_samples = 700

def run(rng, model, strategy, sensor, evaluator, logger):
    while model.num_train < max_num_samples:
        x_new = strategy.get(model=model)
        y_new = sensor.sense(x_new, rng).reshape(-1, 1)
        model.add_data(x_new, y_new)
        model.optimize(num_iter=len(y_new), verbose=False)
        mean, std, error = evaluator.eval_prediction(model)
        logger.append(mean, std, error, x_new, y_new, model.num_train)
        pypolo.experiments.utilities.print_metrics(model, evaluator)

run(rng, model, strategy, sensor, evaluator, logger)

save_dir = "./outputs"
evaluator.save(save_dir)
logger.save(save_dir)
