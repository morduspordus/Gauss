import torch.optim as optim


def get_optimizer(args, model):

    optimizer_type = args['optimizer']

    if optimizer_type == 'Adam':
       optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

    elif optimizer_type == "sgd_trick":
        # https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_tricks.py
        params = [
            {
                "params": [
                    p for name, p in model.named_parameters() if ("bias" in name or "bn" in name)
                ],
                "weight_decay": 0,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if ("bias" not in name and "bn" not in name)
                ]
            },
        ]
        optimizer = optim.SGD(
            params,
            lr=args["learning_rate"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"],
            nesterov=args["nesterov"])

    elif optimizer_type == "sgd_r3":
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if name[-4:] == "bias"
                ],
                "lr": 2 * args["learning_rate"],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if name[-4:] != "bias"
                ],
                "lr": args["learning_rate"],
                "weight_decay": args["weight_decay"],
            },
        ]
        optimizer = optim.SGD(params, momentum=args["momentum"])
    elif optimizer_type == "sgd_all":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
            momentum=args["momentum"],
        )
    else:
        raise NotImplementedError

    # print("optimizer = ", optimizer)
    return optimizer
