
import torch

from data.physgen_dataset import PhysGenDataset


class TripleComponentDataLoader:
    """
    A Dataset Wrapper for loading data with a given batchsize to train 
    Base, Complex and Fusion Components
    """
    def __init__(self, base_dataset, complex_dataset, fusion_dataset, batch_size):
        self.base_dataset = base_dataset
        self.complex_dataset = complex_dataset
        self.fusion_dataset = fusion_dataset
        self.batch_size = batch_size
        self.current_index = 0
        self.last_used_index = 0
        self.dataset_size = min(len(base_dataset), len(complex_dataset), len(fusion_dataset))

    def __len__(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        self.current_index = 0
        while self.current_index < self.dataset_size:
            yield self.get_data(self.current_index, update_index=True)

    def get_data(self, index, update_index=True):
        if index >= self.dataset_size:
            index = 0

        self.last_used_index = index
        # The target index is the current index + the size of the batch or the max size of the dataset
        batch_index = list( range(index, min(index+self.batch_size, self.dataset_size)) )

        # Collect all datapoints seperately
        base_batch = [self.base_dataset[i] for i in batch_index]
        complex_batch = [self.complex_dataset[i] for i in batch_index]
        fusion_batch = [self.fusion_dataset[i] for i in batch_index]

        # Stacking them together to get [BATCH_SIZE, C, H, W]
        base_inputs = torch.stack([item[0] for item in base_batch], dim=0)
        base_targets = torch.stack([item[1] for item in base_batch], dim=0)

        complex_inputs = torch.stack([item[0] for item in complex_batch], dim=0)
        complex_targets = torch.stack([item[1] for item in complex_batch], dim=0)

        fusion_inputs = torch.stack([item[0] for item in fusion_batch], dim=0)
        fusion_targets = torch.stack([item[1] for item in fusion_batch], dim=0)

        if update_index:
            self.current_index = index + self.batch_size

        return ( (base_inputs, base_targets), 
                 (complex_inputs, complex_targets), 
                 (fusion_inputs, fusion_targets),
                  self.last_used_index )
    
    def get_next(self):
        return self.get_data(index=self.current_index, update_index=True)
    
    def get_last(self):
        return self.get_data(index=self.last_used_index, update_index=False)



# Load Basic Datasets
def create_dataloader(opt):
    train_dataset_base = PhysGenDataset(mode='train', variation="sound_baseline", input_type="osm", output_type="standard",
                                                reflexion_channels=opt.reflexion_channels, reflexion_steps=opt.reflexion_steps, reflexions_as_channels=opt.reflexions_as_channels,
                                                reflexions_draw_on_image=opt.reflexions_draw_on_image, force_reflexion_computation=opt.force_reflexion_computation)
    val_dataset_base = PhysGenDataset(mode='validation', variation="sound_baseline", input_type="osm", output_type="standard",
                                            reflexion_channels=opt.reflexion_channels, reflexion_steps=opt.reflexion_steps, reflexions_as_channels=opt.reflexions_as_channels,
                                                reflexions_draw_on_image=opt.reflexions_draw_on_image, force_reflexion_computation=opt.force_reflexion_computation)

    train_dataset_complex = PhysGenDataset(mode='train', variation=opt.variation, input_type="osm", output_type="complex_only",
                                                reflexion_channels=opt.reflexion_channels, reflexion_steps=opt.reflexion_steps, reflexions_as_channels=opt.reflexions_as_channels,
                                                reflexions_draw_on_image=opt.reflexions_draw_on_image, force_reflexion_computation=opt.force_reflexion_computation)
    val_dataset_complex = PhysGenDataset(mode='validation', variation=opt.variation, input_type="osm", output_type="complex_only",
                                                reflexion_channels=opt.reflexion_channels, reflexion_steps=opt.reflexion_steps, reflexions_as_channels=opt.reflexions_as_channels,
                                                reflexions_draw_on_image=opt.reflexions_draw_on_image, force_reflexion_computation=opt.force_reflexion_computation)

    train_dataset_fusion = PhysGenDataset(mode='train', variation=opt.variation, input_type="osm", output_type="standard",
                                                reflexion_channels=opt.reflexion_channels, reflexion_steps=opt.reflexion_steps, reflexions_as_channels=opt.reflexions_as_channels,
                                                reflexions_draw_on_image=opt.reflexions_draw_on_image, force_reflexion_computation=opt.force_reflexion_computation)
    val_dataset_fusion = PhysGenDataset(mode='validation', variation=opt.variation, input_type="osm", output_type="standard",
                                                reflexion_channels=opt.reflexion_channels, reflexion_steps=opt.reflexion_steps, reflexions_as_channels=opt.reflexions_as_channels,
                                                reflexions_draw_on_image=opt.reflexions_draw_on_image, force_reflexion_computation=opt.force_reflexion_computation)

    # Load Dataloader Wrappers
    train_loader = TripleComponentDataLoader(base_dataset=train_dataset_base, 
                                             complex_dataset=train_dataset_complex, 
                                             fusion_dataset=train_dataset_fusion, 
                                             batch_size=opt.batch_size)
    
    val_loader = TripleComponentDataLoader(base_dataset=val_dataset_base, 
                                           complex_dataset=val_dataset_complex, 
                                           fusion_dataset=val_dataset_fusion, 
                                           batch_size=opt.batch_size)
    
    return train_loader, val_loader