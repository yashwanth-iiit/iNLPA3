from src.task1.trainer import train, evaluate_and_save


def main(config_path: str, mode: str):
    if mode == "train":
        train(config_path)
    elif mode == "evaluate":
        evaluate_and_save(config_path)
    elif mode == "both":
        train(config_path)
        evaluate_and_save(config_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
