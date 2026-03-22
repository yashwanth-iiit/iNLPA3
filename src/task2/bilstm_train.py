from src.task2.trainer import train, evaluate_and_save


def main(config_path: str, mode: str):
    if mode == "train":
        train(config_path, model_type="bilstm")
    elif mode == "evaluate":
        evaluate_and_save(config_path, model_type="bilstm")
    elif mode == "both":
        train(config_path, model_type="bilstm")
        evaluate_and_save(config_path, model_type="bilstm")
    else:
        raise ValueError(f"Unknown mode: {mode}")
