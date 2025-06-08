class MuraDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index]['path']
        label = self.dataframe.iloc[index]['target']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).float()

def process_mura_data(image_paths_csv, labeled_studies_csv):
    path_csv = pd.read_csv(image_paths_csv, sep='/', header=None)
    label_csv = pd.read_csv(labeled_studies_csv, sep='/', header=None)

    path_csv[6] = path_csv.apply(lambda row: "/".join(str(x) for x in row), axis=1)
    path_csv.columns = ['folder','set','body_part','patient_id','study_PN','image_id','path']
    label_csv.columns = ['folder','set','body_part','patient_id','study_PN','target']

    label_csv['target'] = label_csv['target'].astype(str).str.replace(',', '').astype(int)

    df = pd.merge(path_csv, label_csv, on=['folder','set','body_part','patient_id','study_PN'])
    return df

def get_transforms(size=256):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, val_transform

def get_dataloaders(train_csv, val_csv, batch_size=64):
    train_df = process_mura_data(
        f"{train_csv}/train_image_paths.csv",
        f"{train_csv}/train_labeled_studies.csv"
    )
    val_df = process_mura_data(
        f"{val_csv}/valid_image_paths.csv",
        f"{val_csv}/valid_labeled_studies.csv"
    )

    train_transform, val_transform = get_transforms()
    train_ds = MuraDataset(train_df, transform=train_transform)
    val_ds   = MuraDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
