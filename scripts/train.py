import dataset
import fruit_dataframe
import warnings
import torch


def create_dataloaders(prep_noiseloader = True, train_batch_size = 2, test_batch_size = 2):

    if prep_noiseloader:
        noise_path = "ScriptDataset/noisy_dataset"
        noise_dataset = dataset.NoiseDataset(noise_path, 100, 512)
        noise_loader = torch.utils.data.DataLoader(noise_dataset, batch_size = test_batch_size, shuffle = True)

    else:
        warnings.warn("Not loading with noise loader, hence Model may generalize worse")
        noise_path = None

    bounding_box_dict, labels_dict = fruit_dataframe.get_dict(["Placeholder", "Apples", "Strawberry", "Apple_Bad_Spot", "Strawberry_Bad_Spot"])
    train_dataset = dataset.FruitDetectDataset(labels_dict, bounding_box_dict, dataset.get_transforms(mode = "train"),
                                        mode = "train", noisy_dataset_path=noise_path)
    test_dataset = dataset.FruitDetectDataset(labels_dict, bounding_box_dict, dataset.get_transforms(mode = "test"), mode = "test",
                                        noisy_dataset_path=noise_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True, collate_fn= collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle = True, collate_fn= collate_fn)

    print("Created Data Loader generators with specified batch size")
    if prep_noiseloader:
        return train_loader, test_loader, noise_loader
    else:
        return train_loader, test_loader

def collate_fn(batch):
  return tuple([list(a) for a in zip(*batch)])

if __name__ == "__main__":
    train_loader, test_loader, noise_loader = create_dataloaders() 
