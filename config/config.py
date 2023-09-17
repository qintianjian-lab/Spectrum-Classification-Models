config = {
    # dev mode
    'debug': False,  # pytorch lightning trainer fast_dev_run
    'wandb_project_name': 'Your Wandb Project Name',
    'enable_wandb': True,  # enable wandb
    # random seed
    'random_seed': 42,
    'used_device': [0],
    'precision': '16-mixed',
    # dataset
    'dataset_dir': 'Your Dataset Dir',
    'spectrum_dir': 'spectrum',
    'label_dir': 'label',
    'type_list': ['class 1', 'class 2', '...'],
    # model
    'used_model': 'sscnn',
    'enable_torch_2.0': False,
    'torch_2.0_compile_mode': 'default',  # default, reduce-overhead, max-autotune
    # hyper-parameters
    'in_channel': 1,
    'spectrum_size': 3584,  # Your Spectrum Size
    'batch_size': 64,
    'num_workers': 4,
    'epochs': 150,
    'learn_rate': 1e-4,
    'cos_annealing_t_0': 30,
    'cos_annealing_t_mult': 2,
    'cos_annealing_eta_min': 1e-6,
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints',
}
