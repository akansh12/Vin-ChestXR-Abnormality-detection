from torchvision import transforms

def vin_big_transform(t_type = 'train'):
    data_transforms = { 
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5), 
            transforms.RandomPerspective(distortion_scale=0.3),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]),

        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])        
        ])

    }
    return data_transforms[t_type]