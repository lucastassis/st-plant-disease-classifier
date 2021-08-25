import torch
import torchvision
import torchvision.transforms as T
from trainer import fit_model, evaluate_model
from loader import split_train_test, get_dataloader

if __name__ == "__main__":

    # define transforms
    train_transforms = T.Compose([T.Resize((224, 224)),
                                  T.ToTensor(),
                                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    
    test_transforms = T.Compose([T.Resize((224, 224)),
                                  T.ToTensor(),
                                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # define dataset
    train_dataset, test_dataset = split_train_test(path_to_data='./dataset/plantvillage/',
                                                   test_split=0.1,
                                                   train_transforms=train_transforms,
                                                   test_transforms=test_transforms)
    # define dataloader
    train_dataloader = get_dataloader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # define model
    net = torchvision.models.mobilenet_v2(pretrained=True)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 8)

    # define optimizer and loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # run training!
    fit_model(model=net, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler, data_loader=train_dataloader, num_epochs=100)






    