import flwr as fl
import torch
from collections import OrderedDict
import os

from model import Restormer
from utils import get_dataloader, calculate_psnr, calculate_ssim, calculate_mse

# Define Flower client
class ImageRestorationClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, valloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Training logic will go here
        print(f"Client {self.cid}: Training for {config['epochs']} epochs")
        # Placeholder for actual training loop
        # For example:
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        # criterion = CharbonnierLoss() # You'll need to define this loss function
        # for epoch in range(config['epochs']):
        #     for input_img, gt_img in self.trainloader:
        #         optimizer.zero_grad()
        #         output = self.model(input_img)
        #         loss = criterion(output, gt_img)
        #         loss.backward()
        #         optimizer.step()
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Evaluation logic will go here
        print(f"Client {self.cid}: Evaluating")
        # Placeholder for actual evaluation loop
        # For example:
        # psnrs = []
        # ssims = []
        # with torch.no_grad():
        #     for input_img, gt_img in self.valloader:
        #         output = self.model(input_img)
        #         psnrs.append(calculate_psnr(output, gt_img))
        #         ssims.append(calculate_ssim(output, gt_img))
        # avg_psnr = sum(psnrs) / len(psnrs)
        # avg_ssim = sum(ssims) / len(ssims)
        loss = 0.0 # Placeholder
        accuracy = 0.0 # Placeholder
        return loss, len(self.valloader.dataset), {"psnr": 0.0, "ssim": 0.0}

# Data loading function
def load_data(client_id):
    base_data_path = "../data"
    train_data_dir = os.path.join(base_data_path, client_id.capitalize(), "Train")
    test_data_dir = os.path.join(base_data_path, client_id.capitalize(), "Test")

    trainloader = get_dataloader(train_data_dir, batch_size=32, train=True)
    valloader = get_dataloader(test_data_dir, batch_size=32, train=False)
    return trainloader, valloader

def load_model():
    # This function will load the Restormer model
    return Restormer() # Instantiate your Restormer model

# Main function to start Flower client
if __name__ == "__main__":
    # Determine client ID based on some logic (e.g., command line argument or environment variable)
    # For demonstration, we'll use a placeholder and assume 3 clients: haze_client, rain_client, snow_client
    # In a real scenario, you would pass this as an argument to the script.
    client_id = os.getenv("CLIENT_ID", "haze_client") # Default to haze_client if not set

    model = load_model()
    trainloader, valloader = load_data(client_id)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=ImageRestorationClient(client_id, model, trainloader, valloader),
    )