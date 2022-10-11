args_resnet50 = { 
    'epochs': 20,  
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'weight_decay': 5e-4   
    },
    'scheduler_name': 'MultiStepLR',
    'scheduler_hyperparameters': {
        'milestones': [10,15],
        'gamma': 0.1
    },
    'batch_size': 256,
}
