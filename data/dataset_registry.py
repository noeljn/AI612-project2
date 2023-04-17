class DatasetRegistry:
    def __init__(self):
        self.datasets = {}

    def register(self, name, dataset_class):
        self.datasets[name] = dataset_class

    def get(self, name):
        return self.datasets.get(name)

# Create a global instance of the DatasetRegistry
dataset_registry = DatasetRegistry()

# Define the register_dataset decorator
def register_dataset(name):
    def decorator(cls):
        dataset_registry.register(name, cls)
        return cls
    return decorator
