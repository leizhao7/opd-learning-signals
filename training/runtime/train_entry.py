import ray
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    try:
        main()
    finally:
        if ray.is_initialized():
            ray.shutdown()
