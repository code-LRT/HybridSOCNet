from src.train import train_model
from src.test import run_test


def main():

    print("Step 1: Training model...")
    train_model()

    print("\nStep 2: Running test and evaluating...")
    run_test()

if __name__ == "__main__":
    main()
