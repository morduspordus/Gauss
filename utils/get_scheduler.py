from torch.optim import lr_scheduler

def get_scheduler(args, optimizer):

    if args['scheduler'] == 'stepLR':
        scheduler = lr_scheduler.StepLR(optimizer, args['scheduler_interval'], args['scheduler_gamma'])
    else:
        print('Scheduler {} not available.'.format(args.optimizer))
        raise NotImplementedError

    return scheduler

