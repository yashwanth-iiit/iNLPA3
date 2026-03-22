from src.task2.trainer import train, evaluate_and_save


def main(config_path: str, mode: str):
    if mode == "train":
        train(config_path, model_type="ssm")
    elif mode == "evaluate":
        evaluate_and_save(config_path, model_type="ssm")
    elif mode == "both":
        train(config_path, model_type="ssm")
        evaluate_and_save(config_path, model_type="ssm")
    else:
        raise ValueError(f"Unknown mode: {mode}")
