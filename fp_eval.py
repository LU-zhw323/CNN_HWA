import torch
import data
import resnet
from utils import evaluate_fp





DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/fp_cnn.th"


def main():

    _, testloader = data.load_cifar10_data(batch_size=50, num_workers=2)

    model = resnet.resnet32().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
   


    avg_loss, accuracy, error_rate = evaluate_fp(model, testloader, DEVICE)
    print(f"Average loss: {avg_loss}, Accuracy: {accuracy}, Error rate: {error_rate}")


if __name__ == "__main__":
    main()