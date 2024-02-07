import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="args for KJS RVSS2024")
    parser.add_argument("--data_folder", type=str, 
                        default="/Users/jy/Documents/PD/research/projects/rvss/train",
                        help="data directory containing all demons")
    
    parser.add_argument("--epochs", type=int, 
                        default=10,
                        help="epochs")

    parser.add_argument("--message", type=str,
                        required=True,
                        help="message for logging the training")
    parser.add_argument("--debug", action="store_true",
                        help="mode for running scripts")
    args = parser.parse_args()
    return args