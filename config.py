import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="args for KJS RVSS2024")
    parser.add_argument("--train_data", type=str, 
                        default="/media/SSD2/Dataset/Self-Driving/train",
                        help="data directory containing all demons")
    parser.add_argument("--test_data", type=str, 
                        default="/media/SSD2/Dataset/Self-Driving/train",
                        help="data directory containing all demons")
    
    parser.add_argument("--epochs", type=int, 
                        default=30,
                        help="epochs")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch_size")

    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="batch_size")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")

    parser.add_argument("--message", type=str,
                        required=True,
                        help="message for logging the training")
    parser.add_argument("--debug", action="store_true",
                        help="mode for running scripts")
    args = parser.parse_args()
    return args